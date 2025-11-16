import torch
import comfy.model_management
import comfy.utils
import node_helpers

class PainterI2VLun:
    """
    An enhanced Wan2.2 Image-to-Video node specifically designed to fix the slow-motion issue in 4-step LoRAs (like lightx2v).
    Now supports faster motion and larger camera movements.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 832, "min": 16, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 4096, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": 4096, "step": 4}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                # 扩大运动幅度范围：1.0-3.0
                "motion_amplitude": ("FLOAT", {"default": 1.8, "min": 1.0, "max": 3.0, "step": 0.1}),
                # 新增：运动加速度（非线性增强）
                "motion_acceleration": ("FLOAT", {"default": 1.2, "min": 1.0, "max": 2.0, "step": 0.1}),
                # 新增：时间维度缩放（增加帧间差异）
                "temporal_scale": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 3.0, "step": 0.1}),
                # 新增：边缘增强强度（让运镜更明显）
                "edge_boost": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
                "start_image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "process"
    CATEGORY = "conditioning/video_models"

    def process(self, positive, negative, vae, width, height, length, batch_size,
                motion_amplitude=1.8, motion_acceleration=1.2, temporal_scale=1.5, 
                edge_boost=0.3, start_image=None, clip_vision_output=None):
        # 1. 严格的零latent初始化（4步LoRA的生命线）
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], 
                           device=comfy.model_management.intermediate_device())
        
        if start_image is not None:
            # 单帧输入处理
            start_image = start_image[:1]
            start_image = comfy.utils.common_upscale(
                start_image.movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
            
            # 创建序列：首帧真实，后续使用渐变灰度（而非固定0.5）
            # 这样可以创造更强的时间差异感
            image = torch.ones((length, height, width, start_image.shape[-1]), 
                             device=start_image.device, dtype=start_image.dtype) * 0.5
            
            # 应用时间维度的渐变（从0.3到0.7的灰度渐变）
            time_gradient = torch.linspace(0.3, 0.7, length, device=start_image.device)
            for i in range(1, length):
                image[i] = image[i] * time_gradient[i]
            
            image[0] = start_image[0]
            
            concat_latent_image = vae.encode(image[:, :, :, :3])
            
            # 单帧mask：仅约束首帧
            mask = torch.ones((1, 1, latent.shape[2], concat_latent_image.shape[-2], 
                             concat_latent_image.shape[-1]), 
                            device=start_image.device, dtype=start_image.dtype)
            mask[:, :, 0] = 0.0
            
            # ========== 核心算法：增强版运动幅度算法 ==========
            if motion_amplitude > 1.0:
                base_latent = concat_latent_image[:, :, 0:1]      # 首帧
                gray_latent = concat_latent_image[:, :, 1:]       # 灰帧序列
                
                # 计算差异
                diff = gray_latent - base_latent
                diff_mean = diff.mean(dim=(1, 3, 4), keepdim=True)
                diff_centered = diff - diff_mean
                
                # === 新增：边缘增强 ===
                # 使用高通滤波增强边缘信息，让运镜更清晰
                if edge_boost > 0:
                    # 简单的高频增强：强化与均值的偏差
                    spatial_mean = diff_centered.mean(dim=(3, 4), keepdim=True)
                    high_freq = diff_centered - spatial_mean
                    diff_centered = diff_centered + high_freq * edge_boost
                
                # === 新增：非线性加速度 ===
                # 对后续帧施加递增的放大系数，模拟加速运动
                num_frames = gray_latent.shape[2]
                acceleration_curve = torch.linspace(1.0, motion_acceleration, num_frames, 
                                                   device=gray_latent.device)
                # 扩展维度以匹配 diff_centered 的形状 [B, C, T, H, W]
                acceleration_curve = acceleration_curve.view(1, 1, -1, 1, 1)
                
                # === 新增：时间维度缩放 ===
                # 放大帧间差异，创造更快的运动感
                diff_centered = diff_centered * temporal_scale
                
                # 应用综合放大：基础幅度 × 加速度曲线
                scaled_latent = base_latent + (diff_centered * motion_amplitude * acceleration_curve) + diff_mean
                
                # === 新增：自适应Clamp ===
                # 根据运动幅度动态调整clamp范围
                clamp_range = min(6.0 + (motion_amplitude - 1.0) * 2, 10.0)
                scaled_latent = torch.clamp(scaled_latent, -clamp_range, clamp_range)
                
                concat_latent_image = torch.cat([base_latent, scaled_latent], dim=2)
            
            # 3. 注入到conditioning
            positive = node_helpers.conditioning_set_values(
                positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
            )
            negative = node_helpers.conditioning_set_values(
                negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
            )

            # 4. 参考帧增强
            ref_latent = vae.encode(start_image[:, :, :, :3])
            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [ref_latent]}, append=True)
            negative = node_helpers.conditioning_set_values(negative, {"reference_latents": [torch.zeros_like(ref_latent)]}, append=True)

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        out_latent = {"samples": latent}
        return (positive, negative, out_latent)


# 节点注册映射
NODE_CLASS_MAPPINGS = {
    "PainterI2VLun": PainterI2VLun,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PainterI2VLun": "PainterI2VLun (Fast Motion Enhanced)",
}