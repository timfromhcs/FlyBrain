import os
import time
import struct
import numpy as np
from typing import Tuple, List, Dict, Optional, Any
try:
    import vulkan as vk
except (ImportError, OSError):  # minimal hosts (e.g. HF Space slim image):
    vk = None  # without the native loader; engine init degrades honestly

from src.paths import resource

class VulkanComputeEngine:
    """
    Persistent Vulkan 1.2+ Compute Engine for biological neural connectome simulation.
    Features:
      - Capability-based device selection (Discrete > Integrated > CPU)
      - Persistent device buffer allocations (no per-step reallocations)
      - GPU-resident neural state (potentials, binary spikes, refractory counters, weights)
      - Persistent descriptor sets and compute pipelines
      - Deterministic synchronization via pipeline barriers
      - High-resolution dispatch latency & throughput telemetry
    """
    def __init__(
        self,
        gpu_index: Optional[int] = None,
        enable_validation: bool = False
    ):
        self.enable_validation = enable_validation
        self.gpu_index = gpu_index
        
        self.instance = None
        self.physical_device = None
        self.device = None
        self.queue = None
        self.queue_family_idx = None
        self.command_pool = None
        self.descriptor_pool = None
        
        self.device_name = "Unknown"
        self.device_type = 0
        self.device_type_str = "Unknown"
        self.driver_version = 0
        self.device_memory_properties = None
        
        # Pipelines
        self.brain_pipeline = None
        self.brain_layout = None
        self.brain_desc_layout = None
        
        self.plasticity_pipeline = None
        self.plasticity_layout = None
        self.plasticity_desc_layout = None
        
        # Persistent Circuit State Resources
        self.loaded_circuit_hash = None
        self.num_neurons = 0
        self.num_synapses = 0
        self.persistent_buffers = {}
        self.brain_descriptor_set = None
        self.plasticity_descriptor_set = None
        self.brain_command_buffer = None
        self.plasticity_command_buffer = None
        
        # Diagnostics
        self.last_step_latency_ms = 0.0
        self.last_plasticity_latency_ms = 0.0
        self.total_steps_executed = 0

        if vk is None:
            raise RuntimeError("Vulkan bindings unavailable on this host "
                               "(no native loader); CPU reference is the honest fallback.")
        self._init_vulkan()

    def _find_memory_type(self, type_filter: int, properties: int) -> int:
        for i in range(self.device_memory_properties.memoryTypeCount):
            if (type_filter & (1 << i)) and (self.device_memory_properties.memoryTypes[i].propertyFlags & properties) == properties:
                return i
        raise RuntimeError("Failed to find suitable memory type!")

    def _init_vulkan(self):
        app_info = vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName="FlyBrainVulkanEngine",
            applicationVersion=vk.VK_MAKE_VERSION(1, 1, 0),
            pEngineName="FlyBrain",
            engineVersion=vk.VK_MAKE_VERSION(1, 1, 0),
            apiVersion=vk.VK_MAKE_VERSION(1, 2, 0),
        )

        layers = []
        if self.enable_validation:
            layers.append("VK_LAYER_KHRONOS_validation")

        create_info = vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=app_info,
            enabledExtensionCount=0,
            ppEnabledExtensionNames=[],
            enabledLayerCount=len(layers),
            ppEnabledLayerNames=layers,
        )
        self.instance = vk.vkCreateInstance(create_info, None)

        # 1. Capability-based physical device selection
        pdevs = vk.vkEnumeratePhysicalDevices(self.instance)
        if not pdevs:
            raise RuntimeError("No Vulkan physical devices found!")

        selected_dev = None
        selected_q_fam = None
        selected_props = None

        # Prioritize: DISCRETE_GPU (2) > INTEGRATED_GPU (1) > CPU (4)
        scored_devices = []
        for idx, dev in enumerate(pdevs):
            props = vk.vkGetPhysicalDeviceProperties(dev)
            q_fam_props = vk.vkGetPhysicalDeviceQueueFamilyProperties(dev)
            compute_fam = -1
            for q_idx, q_fam in enumerate(q_fam_props):
                if q_fam.queueFlags & vk.VK_QUEUE_COMPUTE_BIT:
                    compute_fam = q_idx
                    break
            
            if compute_fam == -1:
                continue

            type_score = (
                100 if props.deviceType == vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU else
                50 if props.deviceType == vk.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU else
                10 if props.deviceType == vk.VK_PHYSICAL_DEVICE_TYPE_CPU else 1
            )
            scored_devices.append((type_score, idx, dev, compute_fam, props))

        if not scored_devices:
            raise RuntimeError("No compute-capable Vulkan physical devices found!")

        scored_devices.sort(key=lambda x: x[0], reverse=True)

        if self.gpu_index is not None and 0 <= self.gpu_index < len(pdevs):
            # Explicit index selection
            for item in scored_devices:
                if item[1] == self.gpu_index:
                    _, _, selected_dev, selected_q_fam, selected_props = item
                    break
        if selected_dev is None:
            _, _, selected_dev, selected_q_fam, selected_props = scored_devices[0]

        self.physical_device = selected_dev
        self.queue_family_idx = selected_q_fam
        self.device_name = selected_props.deviceName
        self.device_type = selected_props.deviceType
        self.device_type_str = (
            "Discrete GPU" if selected_props.deviceType == vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU else
            "Integrated GPU" if selected_props.deviceType == vk.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU else
            "CPU" if selected_props.deviceType == vk.VK_PHYSICAL_DEVICE_TYPE_CPU else "Other"
        )
        self.driver_version = selected_props.driverVersion
        self.device_memory_properties = vk.vkGetPhysicalDeviceMemoryProperties(self.physical_device)

        # 2. Logical Device Creation
        q_create = vk.VkDeviceQueueCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            queueFamilyIndex=self.queue_family_idx,
            queueCount=1,
            pQueuePriorities=[1.0],
        )

        d_create = vk.VkDeviceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            queueCreateInfoCount=1,
            pQueueCreateInfos=[q_create],
            enabledExtensionCount=0,
            ppEnabledExtensionNames=[],
            pEnabledFeatures=None,
        )
        self.device = vk.vkCreateDevice(self.physical_device, d_create, None)
        self.queue = vk.vkGetDeviceQueue(self.device, self.queue_family_idx, 0)

        # 3. Command Pool
        cmd_pool_info = vk.VkCommandPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            queueFamilyIndex=self.queue_family_idx,
            flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        )
        self.command_pool = vk.vkCreateCommandPool(self.device, cmd_pool_info, None)
        
        # 4. Descriptor Pool
        pool_sizes = [
            vk.VkDescriptorPoolSize(
                type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=2048,
            )
        ]
        pool_info = vk.VkDescriptorPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            flags=vk.VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
            maxSets=256,
            poolSizeCount=len(pool_sizes),
            pPoolSizes=pool_sizes,
        )
        self.descriptor_pool = vk.vkCreateDescriptorPool(self.device, pool_info, None)

        # 5. Build Compute Pipelines
        self._init_brain_pipeline()
        self._init_plasticity_pipeline()

    def _create_buffer(self, size_bytes: int, usage: int):
        buf_info = vk.VkBufferCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=size_bytes,
            usage=usage,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
        )
        buf = vk.vkCreateBuffer(self.device, buf_info, None)
        reqs = vk.vkGetBufferMemoryRequirements(self.device, buf)
        mem_type_idx = self._find_memory_type(
            reqs.memoryTypeBits,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        )
        alloc_info = vk.VkMemoryAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=reqs.size,
            memoryTypeIndex=mem_type_idx,
        )
        mem = vk.vkAllocateMemory(self.device, alloc_info, None)
        vk.vkBindBufferMemory(self.device, buf, mem, 0)
        return buf, mem, reqs.size

    def _init_brain_pipeline(self):
        spv_path = resource("shaders/brain_step.spv")
        if not os.path.exists(spv_path):
            raise FileNotFoundError(f"SPIR-V compute shader missing: {spv_path}")
        with open(spv_path, "rb") as f:
            code = f.read()

        mod_info = vk.VkShaderModuleCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            codeSize=len(code),
            pCode=code,
        )
        shader_module = vk.vkCreateShaderModule(self.device, mod_info, None)

        # 11 bindings for LIF:
        # 0: RowOffsets, 1: ColIndices, 2: Weights, 3: PrevSpikes, 4: ExternalInputs,
        # 5: PotentialsIn, 6: RefractoryIn, 7: PotentialsOut, 8: SpikesOut, 9: RefractoryOut, 10: Params
        bindings = []
        for b_idx in range(11):
            bindings.append(vk.VkDescriptorSetLayoutBinding(
                binding=b_idx,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ))

        layout_info = vk.VkDescriptorSetLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount=len(bindings),
            pBindings=bindings,
        )
        self.brain_desc_layout = vk.vkCreateDescriptorSetLayout(self.device, layout_info, None)

        pipe_layout_info = vk.VkPipelineLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=1,
            pSetLayouts=[self.brain_desc_layout],
            pushConstantRangeCount=0,
            pPushConstantRanges=[],
        )
        self.brain_layout = vk.vkCreatePipelineLayout(self.device, pipe_layout_info, None)

        stage_info = vk.VkPipelineShaderStageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            module=shader_module,
            pName="main",
        )

        pipe_create_info = vk.VkComputePipelineCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            stage=stage_info,
            layout=self.brain_layout,
        )
        self.brain_pipeline = vk.vkCreateComputePipelines(self.device, vk.VK_NULL_HANDLE, 1, [pipe_create_info], None)[0]
        vk.vkDestroyShaderModule(self.device, shader_module, None)

    def _init_plasticity_pipeline(self):
        spv_path = resource("shaders/plasticity.spv")
        if not os.path.exists(spv_path):
            raise FileNotFoundError(f"SPIR-V compute shader missing: {spv_path}")
        with open(spv_path, "rb") as f:
            code = f.read()

        mod_info = vk.VkShaderModuleCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            codeSize=len(code),
            pCode=code,
        )
        shader_module = vk.vkCreateShaderModule(self.device, mod_info, None)

        bindings = []
        for b_idx in range(6):
            bindings.append(vk.VkDescriptorSetLayoutBinding(
                binding=b_idx,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ))

        layout_info = vk.VkDescriptorSetLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount=len(bindings),
            pBindings=bindings,
        )
        self.plasticity_desc_layout = vk.vkCreateDescriptorSetLayout(self.device, layout_info, None)

        pipe_layout_info = vk.VkPipelineLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=1,
            pSetLayouts=[self.plasticity_desc_layout],
            pushConstantRangeCount=0,
            pPushConstantRanges=[],
        )
        self.plasticity_layout = vk.vkCreatePipelineLayout(self.device, pipe_layout_info, None)

        stage_info = vk.VkPipelineShaderStageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            module=shader_module,
            pName="main",
        )

        pipe_create_info = vk.VkComputePipelineCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            stage=stage_info,
            layout=self.plasticity_layout,
        )
        self.plasticity_pipeline = vk.vkCreateComputePipelines(self.device, vk.VK_NULL_HANDLE, 1, [pipe_create_info], None)[0]
        vk.vkDestroyShaderModule(self.device, shader_module, None)

    def load_circuit(
        self,
        row_offsets: np.ndarray,
        col_indices: np.ndarray,
        weights: np.ndarray,
        initial_potentials: Optional[np.ndarray] = None,
        initial_spikes: Optional[np.ndarray] = None,
        initial_refractory: Optional[np.ndarray] = None
    ):
        """
        Allocates persistent GPU device buffers and writes persistent descriptor sets.
        Avoids all per-step buffer allocations.
        """
        self.free_circuit_resources()
        
        N = len(row_offsets) - 1
        M = len(col_indices)
        self.num_neurons = N
        self.num_synapses = M
        
        row_offsets = np.ascontiguousarray(row_offsets, dtype=np.int32)
        col_indices = np.ascontiguousarray(col_indices, dtype=np.int32)
        weights = np.ascontiguousarray(weights, dtype=np.float32)
        
        if initial_potentials is None:
            initial_potentials = np.zeros(N, dtype=np.float32)
        if initial_spikes is None:
            initial_spikes = np.zeros(N, dtype=np.float32)
        if initial_refractory is None:
            initial_refractory = np.zeros(N, dtype=np.int32)
            
        initial_potentials = np.ascontiguousarray(initial_potentials, dtype=np.float32)
        initial_spikes = np.ascontiguousarray(initial_spikes, dtype=np.float32)
        initial_ext = np.zeros(N, dtype=np.float32)
        initial_ref = np.ascontiguousarray(initial_refractory, dtype=np.int32)
        
        # 1. Allocate persistent buffers
        def alloc_and_upload(name: str, arr: np.ndarray):
            buf, mem, sz = self._create_buffer(arr.nbytes, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
            ptr = vk.vkMapMemory(self.device, mem, 0, arr.nbytes, 0)
            ptr[0:arr.nbytes] = arr.tobytes()
            vk.vkUnmapMemory(self.device, mem)
            self.persistent_buffers[name] = {"buf": buf, "mem": mem, "size": sz, "bytes": arr.nbytes}

        alloc_and_upload("row_offsets", row_offsets)
        alloc_and_upload("col_indices", col_indices)
        alloc_and_upload("weights", weights)
        alloc_and_upload("prev_spikes", initial_spikes)
        alloc_and_upload("ext_inputs", initial_ext)
        alloc_and_upload("potentials_in", initial_potentials)
        alloc_and_upload("refractory_in", initial_ref)
        alloc_and_upload("potentials_out", initial_potentials)
        alloc_and_upload("spikes_out", initial_spikes)
        alloc_and_upload("refractory_out", initial_ref)
        
        # Params buffer: N, decay, threshold, v_reset, v_rest, t_ref
        params_bytes = struct.pack('iffffi', N, 0.85, 1.0, 0.0, 0.0, 2)
        p_buf, p_mem, p_sz = self._create_buffer(len(params_bytes), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        ptr = vk.vkMapMemory(self.device, p_mem, 0, len(params_bytes), 0)
        ptr[0:len(params_bytes)] = params_bytes
        vk.vkUnmapMemory(self.device, p_mem)
        self.persistent_buffers["brain_params"] = {"buf": p_buf, "mem": p_mem, "size": p_sz, "bytes": len(params_bytes)}

        # Plasticity params buffer: M, N, lr, reward, weight_decay, min_w, max_w
        plas_bytes = struct.pack('iifffff', M, N, 0.05, 0.0, 0.01, 0.01, 1.0)
        pl_buf, pl_mem, pl_sz = self._create_buffer(len(plas_bytes), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        self.persistent_buffers["plasticity_params"] = {"buf": pl_buf, "mem": pl_mem, "size": pl_sz, "bytes": len(plas_bytes)}

        # 2. Allocate persistent descriptor set for brain step
        b_alloc_info = vk.VkDescriptorSetAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool=self.descriptor_pool,
            descriptorSetCount=1,
            pSetLayouts=[self.brain_desc_layout],
        )
        self.brain_descriptor_set = vk.vkAllocateDescriptorSets(self.device, b_alloc_info)[0]

        # Bindings 0..10
        ordered_keys = [
            "row_offsets", "col_indices", "weights", "prev_spikes", "ext_inputs",
            "potentials_in", "refractory_in", "potentials_out", "spikes_out", "refractory_out", "brain_params"
        ]
        writes = []
        for b_idx, key in enumerate(ordered_keys):
            b_item = self.persistent_buffers[key]
            b_info = vk.VkDescriptorBufferInfo(buffer=b_item["buf"], offset=0, range=b_item["size"])
            writes.append(vk.VkWriteDescriptorSet(
                sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=self.brain_descriptor_set,
                dstBinding=b_idx,
                dstArrayElement=0,
                descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=[b_info],
            ))
        vk.vkUpdateDescriptorSets(self.device, len(writes), writes, 0, None)

        # 3. Allocate persistent descriptor set for plasticity step
        # Bindings 0..5: RowOffsets, ColIndices, Weights, SpikesOut (post), PrevSpikes (pre), PlasticityParams
        p_alloc_info = vk.VkDescriptorSetAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool=self.descriptor_pool,
            descriptorSetCount=1,
            pSetLayouts=[self.plasticity_desc_layout],
        )
        self.plasticity_descriptor_set = vk.vkAllocateDescriptorSets(self.device, p_alloc_info)[0]

        plas_keys = ["row_offsets", "col_indices", "weights", "spikes_out", "prev_spikes", "plasticity_params"]
        plas_writes = []
        for b_idx, key in enumerate(plas_keys):
            b_item = self.persistent_buffers[key]
            b_info = vk.VkDescriptorBufferInfo(buffer=b_item["buf"], offset=0, range=b_item["size"])
            plas_writes.append(vk.VkWriteDescriptorSet(
                sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=self.plasticity_descriptor_set,
                dstBinding=b_idx,
                dstArrayElement=0,
                descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=[b_info],
            ))
        vk.vkUpdateDescriptorSets(self.device, len(plas_writes), plas_writes, 0, None)

        # 4. Allocate persistent command buffer
        cb_alloc = vk.VkCommandBufferAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=self.command_pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        )
        self.brain_command_buffer = vk.vkAllocateCommandBuffers(self.device, cb_alloc)[0]

    def run_step_persistent(
        self,
        external_inputs: Optional[np.ndarray] = None,
        decay: float = 0.85,
        threshold: float = 1.0,
        v_reset: float = 0.0,
        v_rest: float = 0.0,
        t_ref: int = 2,
        readback: bool = True
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Executes one LIF simulation step entirely on GPU using persistent resources.
        Zero buffer allocation or descriptor reallocation.
        Returns authoritative GPU state (potentials, spikes, refractory) — callers
        must use the returned refractory counters, never recompute them locally.
        """
        if not self.persistent_buffers:
            raise RuntimeError("No circuit loaded in Vulkan compute engine!")

        N = self.num_neurons
        t0 = time.perf_counter_ns()

        # Update external inputs if provided
        if external_inputs is not None:
            ext_bytes = np.ascontiguousarray(external_inputs, dtype=np.float32).tobytes()
            ext_mem = self.persistent_buffers["ext_inputs"]["mem"]
            ptr = vk.vkMapMemory(self.device, ext_mem, 0, len(ext_bytes), 0)
            ptr[0:len(ext_bytes)] = ext_bytes
            vk.vkUnmapMemory(self.device, ext_mem)

        # Update params struct
        p_bytes = struct.pack('iffffi', N, decay, threshold, v_reset, v_rest, t_ref)
        p_mem = self.persistent_buffers["brain_params"]["mem"]
        ptr = vk.vkMapMemory(self.device, p_mem, 0, len(p_bytes), 0)
        ptr[0:len(p_bytes)] = p_bytes
        vk.vkUnmapMemory(self.device, p_mem)

        # Record command buffer
        cmd = self.brain_command_buffer
        vk.vkResetCommandBuffer(cmd, 0)
        
        begin_info = vk.VkCommandBufferBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        )
        vk.vkBeginCommandBuffer(cmd, begin_info)
        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.brain_pipeline)
        vk.vkCmdBindDescriptorSets(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.brain_layout, 0, 1, [self.brain_descriptor_set], 0, None)
        
        group_count = (N + 63) // 64
        vk.vkCmdDispatch(cmd, group_count, 1, 1)

        # Pipeline barrier to ensure outputs are written before copying for next step
        barrier = vk.VkMemoryBarrier(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
            dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_HOST_READ_BIT,
        )
        vk.vkCmdPipelineBarrier(
            cmd,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | vk.VK_PIPELINE_STAGE_HOST_BIT,
            0,
            1, [barrier],
            0, None,
            0, None
        )
        
        # Ping-pong copy on GPU: potentials_out -> potentials_in, spikes_out -> prev_spikes, refractory_out -> refractory_in
        copy_pot = vk.VkBufferCopy(srcOffset=0, dstOffset=0, size=N * 4)
        vk.vkCmdCopyBuffer(cmd, self.persistent_buffers["potentials_out"]["buf"], self.persistent_buffers["potentials_in"]["buf"], 1, [copy_pot])
        vk.vkCmdCopyBuffer(cmd, self.persistent_buffers["spikes_out"]["buf"], self.persistent_buffers["prev_spikes"]["buf"], 1, [copy_pot])
        vk.vkCmdCopyBuffer(cmd, self.persistent_buffers["refractory_out"]["buf"], self.persistent_buffers["refractory_in"]["buf"], 1, [copy_pot])

        vk.vkEndCommandBuffer(cmd)

        submit_info = vk.VkSubmitInfo(
            sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1,
            pCommandBuffers=[cmd],
        )
        vk.vkQueueSubmit(self.queue, 1, [submit_info], vk.VK_NULL_HANDLE)
        vk.vkQueueWaitIdle(self.queue)

        t1 = time.perf_counter_ns()
        self.last_step_latency_ms = (t1 - t0) / 1_000_000.0
        self.total_steps_executed += 1

        if readback:
            # Read back state from host-coherent buffer
            pot_mem = self.persistent_buffers["potentials_out"]["mem"]
            spk_mem = self.persistent_buffers["spikes_out"]["mem"]
            ref_mem = self.persistent_buffers["refractory_out"]["mem"]

            ptr = vk.vkMapMemory(self.device, pot_mem, 0, N * 4, 0)
            pot_out = np.frombuffer(bytes(ptr[0:N * 4]), dtype=np.float32).copy()
            vk.vkUnmapMemory(self.device, pot_mem)

            ptr = vk.vkMapMemory(self.device, spk_mem, 0, N * 4, 0)
            spk_out = np.frombuffer(bytes(ptr[0:N * 4]), dtype=np.float32).copy()
            vk.vkUnmapMemory(self.device, spk_mem)

            ptr = vk.vkMapMemory(self.device, ref_mem, 0, N * 4, 0)
            ref_out = np.frombuffer(bytes(ptr[0:N * 4]), dtype=np.int32).copy()
            vk.vkUnmapMemory(self.device, ref_mem)
            return pot_out, spk_out, ref_out

        return None, None, None

    def run_plasticity_persistent(
        self,
        learning_rate: float = 0.05,
        reward: float = 1.0,
        weight_decay: float = 0.01,
        min_weight: float = 0.01,
        max_weight: float = 1.0,
        readback: bool = False
    ) -> Optional[np.ndarray]:
        """
        Executes synaptic plasticity step on GPU-resident weights in-place.
        """
        if not self.persistent_buffers:
            return None

        M = self.num_synapses
        t0 = time.perf_counter_ns()

        # Update plasticity parameters
        M = self.num_synapses
        N = self.num_neurons
        pl_bytes = struct.pack('iifffff', M, N, learning_rate, reward, weight_decay, min_weight, max_weight)
        pl_mem = self.persistent_buffers["plasticity_params"]["mem"]
        ptr = vk.vkMapMemory(self.device, pl_mem, 0, len(pl_bytes), 0)
        ptr[0:len(pl_bytes)] = pl_bytes
        vk.vkUnmapMemory(self.device, pl_mem)

        cmd = self.brain_command_buffer
        vk.vkResetCommandBuffer(cmd, 0)
        begin_info = vk.VkCommandBufferBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        )
        vk.vkBeginCommandBuffer(cmd, begin_info)
        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.plasticity_pipeline)
        vk.vkCmdBindDescriptorSets(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.plasticity_layout, 0, 1, [self.plasticity_descriptor_set], 0, None)
        
        group_count = (M + 63) // 64
        vk.vkCmdDispatch(cmd, group_count, 1, 1)

        barrier = vk.VkMemoryBarrier(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
            dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_HOST_READ_BIT,
        )
        vk.vkCmdPipelineBarrier(
            cmd,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | vk.VK_PIPELINE_STAGE_HOST_BIT,
            0,
            1, [barrier],
            0, None,
            0, None
        )
        vk.vkEndCommandBuffer(cmd)

        submit_info = vk.VkSubmitInfo(
            sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1,
            pCommandBuffers=[cmd],
        )
        vk.vkQueueSubmit(self.queue, 1, [submit_info], vk.VK_NULL_HANDLE)
        vk.vkQueueWaitIdle(self.queue)

        t1 = time.perf_counter_ns()
        self.last_plasticity_latency_ms = (t1 - t0) / 1_000_000.0

        if readback:
            w_mem = self.persistent_buffers["weights"]["mem"]
            ptr = vk.vkMapMemory(self.device, w_mem, 0, M * 4, 0)
            weights_out = np.frombuffer(bytes(ptr[0:M * 4]), dtype=np.float32).copy()
            vk.vkUnmapMemory(self.device, w_mem)
            return weights_out

        return None

    def download_weights(self) -> Optional[np.ndarray]:
        """Pure GPU->CPU weight readback with no dispatch (sync primitive).

        Used for lazy weight synchronization: GPU weights stay authoritative
        across steps; the CPU mirror is refreshed only at explicit sync points
        (snapshots, experiment manifests, validation), never per step.
        """
        if not self.persistent_buffers:
            return None
        import numpy as _np
        M = self.num_synapses
        w_mem = self.persistent_buffers["weights"]["mem"]
        ptr = vk.vkMapMemory(self.device, w_mem, 0, M * 4, 0)
        out = _np.frombuffer(bytes(ptr[0:M * 4]), dtype=_np.float32).copy()
        vk.vkUnmapMemory(self.device, w_mem)
        return out

    def upload_buffer_data(self, name: str, arr: np.ndarray):
        """Uploads contiguous numpy array data to an existing persistent GPU buffer."""
        if name in self.persistent_buffers:
            b_item = self.persistent_buffers[name]
            arr_bytes = np.ascontiguousarray(arr).tobytes()
            copy_len = min(len(arr_bytes), b_item["bytes"])
            ptr = vk.vkMapMemory(self.device, b_item["mem"], 0, copy_len, 0)
            ptr[0:copy_len] = arr_bytes[:copy_len]
            vk.vkUnmapMemory(self.device, b_item["mem"])

    def run_step(
        self,
        row_offsets: np.ndarray,
        col_indices: np.ndarray,
        weights: np.ndarray,
        prev_activations: np.ndarray,
        external_inputs: np.ndarray,
        potentials_in: np.ndarray,
        decay: float = 0.85,
        threshold: float = 1.0,
        leak: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Legacy / test harness compatibility method:
        Loads circuit if not loaded, updates persistent buffers with provided inputs, executes step, and returns (potentials, spikes).
        """
        N = len(potentials_in)
        if (not self.persistent_buffers) or (self.num_neurons != N) or (self.num_synapses != len(weights)):
            self.load_circuit(row_offsets, col_indices, weights, potentials_in, prev_activations)
        else:
            self.upload_buffer_data("weights", weights)
            self.upload_buffer_data("prev_spikes", prev_activations)
            self.upload_buffer_data("potentials_in", potentials_in)
            self.upload_buffer_data("refractory_in", np.zeros(N, dtype=np.int32))
            
        pot_out, spk_out, _ = self.run_step_persistent(
            external_inputs=external_inputs,
            decay=decay,
            threshold=threshold,
            readback=True
        )
        return pot_out, spk_out

    def run_plasticity_step(
        self,
        row_offsets: np.ndarray,
        col_indices: np.ndarray,
        weights: np.ndarray,
        post_activations: np.ndarray,
        pre_activations: np.ndarray,
        learning_rate: float = 0.05,
        reward: float = 1.0,
        weight_decay: float = 0.01,
        min_weight: float = 0.01,
        max_weight: float = 1.0
    ) -> np.ndarray:
        """
        Legacy / test harness compatibility method:
        Executes plasticity step and returns updated weights.
        """
        if (not self.persistent_buffers) or (self.num_synapses != len(weights)):
            self.load_circuit(row_offsets, col_indices, weights, None, pre_activations)
        else:
            self.upload_buffer_data("weights", weights)
            self.upload_buffer_data("prev_spikes", pre_activations)
            self.upload_buffer_data("spikes_out", post_activations)
            
        w_out = self.run_plasticity_persistent(
            learning_rate=learning_rate,
            reward=reward,
            weight_decay=weight_decay,
            min_weight=min_weight,
            max_weight=max_weight,
            readback=True
        )
        return w_out if w_out is not None else weights

    def get_diagnostics(self) -> Dict[str, Any]:
        """Returns real live hardware diagnostics and performance telemetry."""
        total_mem_allocated = sum(b["bytes"] for b in self.persistent_buffers.values()) if self.persistent_buffers else 0
        steps_per_sec = round(1000.0 / max(0.001, self.last_step_latency_ms), 1) if self.last_step_latency_ms > 0 else 0.0
        neurons_per_sec = int(steps_per_sec * self.num_neurons)
        synapses_per_sec = int(steps_per_sec * self.num_synapses)

        return {
            "device_name": self.device_name,
            "device_type": int(self.device_type),
            "device_type_str": self.device_type_str,
            "driver_version": int(self.driver_version),
            "queue_family_idx": self.queue_family_idx,
            "total_gpu_memory_allocated_bytes": total_mem_allocated,
            "step_latency_ms": round(self.last_step_latency_ms, 3),
            "plasticity_latency_ms": round(self.last_plasticity_latency_ms, 3),
            "throughput_steps_per_sec": steps_per_sec,
            "throughput_neurons_per_sec": neurons_per_sec,
            "throughput_synapses_per_sec": synapses_per_sec,
            "total_steps_executed": self.total_steps_executed,
            "status": "OPERATIONAL"
        }

    def free_circuit_resources(self):
        """Frees persistent circuit buffers and descriptor sets."""
        if self.device is None:
            return
            
        if self.brain_command_buffer is not None:
            vk.vkFreeCommandBuffers(self.device, self.command_pool, 1, [self.brain_command_buffer])
            self.brain_command_buffer = None

        if self.brain_descriptor_set is not None:
            vk.vkFreeDescriptorSets(self.device, self.descriptor_pool, 1, [self.brain_descriptor_set])
            self.brain_descriptor_set = None

        if self.plasticity_descriptor_set is not None:
            vk.vkFreeDescriptorSets(self.device, self.descriptor_pool, 1, [self.plasticity_descriptor_set])
            self.plasticity_descriptor_set = None

        for name, item in self.persistent_buffers.items():
            vk.vkDestroyBuffer(self.device, item["buf"], None)
            vk.vkFreeMemory(self.device, item["mem"], None)
        self.persistent_buffers.clear()

    def cleanup(self):
        """Full cleanup of all Vulkan resources."""
        if self.device is None:
            return
        vk.vkDeviceWaitIdle(self.device)
        self.free_circuit_resources()

        if self.descriptor_pool:
            vk.vkDestroyDescriptorPool(self.device, self.descriptor_pool, None)
            self.descriptor_pool = None

        if self.brain_pipeline:
            vk.vkDestroyPipeline(self.device, self.brain_pipeline, None)
            self.brain_pipeline = None
        if self.brain_layout:
            vk.vkDestroyPipelineLayout(self.device, self.brain_layout, None)
            self.brain_layout = None
        if self.brain_desc_layout:
            vk.vkDestroyDescriptorSetLayout(self.device, self.brain_desc_layout, None)
            self.brain_desc_layout = None

        if self.plasticity_pipeline:
            vk.vkDestroyPipeline(self.device, self.plasticity_pipeline, None)
            self.plasticity_pipeline = None
        if self.plasticity_layout:
            vk.vkDestroyPipelineLayout(self.device, self.plasticity_layout, None)
            self.plasticity_layout = None
        if self.plasticity_desc_layout:
            vk.vkDestroyDescriptorSetLayout(self.device, self.plasticity_desc_layout, None)
            self.plasticity_desc_layout = None

        if self.command_pool:
            vk.vkDestroyCommandPool(self.device, self.command_pool, None)
            self.command_pool = None

        if self.device:
            vk.vkDestroyDevice(self.device, None)
            self.device = None

        if self.instance:
            vk.vkDestroyInstance(self.instance, None)
            self.instance = None

VulkanBrainBackend = VulkanComputeEngine
