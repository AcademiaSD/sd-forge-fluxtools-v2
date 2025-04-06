#v023
from backend import memory_management
from backend.diffusion_engine.flux import Flux
import gradio
from gradio_rangeslider import RangeSlider
import torch, numpy as np
from modules import scripts, shared
from modules.ui_components import InputAccordion
from modules.script_callbacks import on_cfg_denoiser, remove_current_script_callbacks
from modules.sd_samplers_common import images_tensor_to_samples, approximation_indexes
from modules_forge.forge_canvas.canvas import ForgeCanvas
from PIL import Image, ImageDraw
from controlnet_aux import CannyDetector
from controlnet_aux.processor import Processor

import gc
from modules_forge import main_entry


class flux1tools(scripts.Script):
    sorting_priority = 0

    glc_backup_flux = None
    clearConds = False
    text_encoder_device_backup = None
    sigmasBackup = None  # Se conserva por si se requiere en ajustes futuros

    flux_use_T5 = True
    flux_use_CL = True

    def __init__(self):
        if flux1tools.glc_backup_flux is None:
            flux1tools.glc_backup_flux = Flux.get_learned_conditioning
        if flux1tools.text_encoder_device_backup is None:
            flux1tools.text_encoder_device_backup = memory_management.text_encoder_device

    # Función auxiliar para separar el prompt en 2 partes usando "SPLIT"
    def splitPrompt(prompt, countTextEncoders):
        promptTE1 = []
        promptTE2 = []
        for p in prompt:
            parts = p.split('SPLIT')
            if len(parts) >= 2:
                promptTE1.append(parts[0].strip())
                promptTE2.append(parts[1].strip())
            else:
                promptTE1.append(p)
                promptTE2.append(p)
        return promptTE1, promptTE2

    # Funciones para seleccionar el dispositivo de los text encoders
    def patched_text_encoder_gpu2():
        if torch.cuda.device_count() > 1:
            return torch.device("cuda:1")
        else:
            return torch.cuda.current_device()
    def patched_text_encoder_gpu():
        return torch.cuda.current_device()
    def patched_text_encoder_cpu():
        return memory_management.cpu

    @torch.inference_mode()
    def patched_glc_flux(self, prompt: list[str]):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)
        nprompt = len(prompt)
        CLIPprompt, T5prompt = flux1tools.splitPrompt(prompt, 2)
        if flux1tools.flux_use_CL:
            cond_l, pooled_l = self.text_processing_engine_l(CLIPprompt)
        else:
            pooled_l = torch.zeros([nprompt, 768])
        if flux1tools.flux_use_T5:
            cond_t5 = self.text_processing_engine_t5(prompt)
        else:
            cond_t5 = torch.zeros([nprompt, 256, 4096])
        cond = dict(crossattn=cond_t5, vector=pooled_l)
        if getattr(self, "use_distilled_cfg_scale", False):
            distilled_cfg_scale = getattr(prompt, 'distilled_cfg_scale', 3.5) or 3.5
            cond['guidance'] = torch.FloatTensor([distilled_cfg_scale] * len(prompt))
            print(f'Distilled CFG Scale: {distilled_cfg_scale}')
        else:
            print('Distilled CFG Scale will be ignored for Schnell')
        return cond

    def title(self):
        return "Flux1Tools-V2"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    # --- Funciones de preprocesamiento ---
    def run_canny_preprocessor(self, image, low_threshold, high_threshold, detect_resolution, image_resolution):
        try:
            if image is None:
                return None
            if not hasattr(image, "size"):
                image = Image.fromarray(np.array(image))
            if image.mode != "RGB":
                image = image.convert("RGB")
            processor = CannyDetector()
            processed = processor(
                image,
                low_threshold=int(low_threshold),
                high_threshold=int(high_threshold),
                detect_resolution=int(detect_resolution),
                image_resolution=int(image_resolution)
            )
            return processed
        except Exception as e:
            print("Error in run_canny_preprocessor:", e)
            return None

    def run_depth_preprocessor(self, image, processor_id):
        try:
            if image is None:
                return None
            if not hasattr(image, "size"):
                image = Image.fromarray(np.array(image))
            if image.mode != "RGB":
                image = image.convert("RGB")
            proc = Processor(processor_id)
            processed = proc(image)
            processed = processed.resize((512, 512), Image.BICUBIC)
            return processed
        except Exception as e:
            print("Error in run_depth_preprocessor:", e)
            return None

    # --- Nueva función para expandir lienzo (outpaint) ---
    def expand_canvas(self, image, mask, expand_up, expand_down, expand_left, expand_right):
        try:
            if image is None:
                return None, None
                
            # Convertir a PIL Image si es necesario
            if not hasattr(image, "size"):
                image = Image.fromarray(np.array(image))
            if not hasattr(mask, "size"):
                mask = Image.fromarray(np.array(mask))
                
            # Obtener dimensiones originales
            orig_width, orig_height = image.size
            
            # Calcular nuevas dimensiones
            new_width = orig_width + expand_left + expand_right
            new_height = orig_height + expand_up + expand_down
            
            # Crear nueva imagen y máscara expandidas
            new_image = Image.new("RGBA", (new_width, new_height), (0, 0, 0, 0))
            new_mask = Image.new("RGBA", (new_width, new_height), (0, 0, 0, 0))
            
            # Colocar imagen original en la nueva imagen
            new_image.paste(image, (expand_left, expand_up))
            
            # Crear máscara para areas expandidas (áreas a generar)
            draw = ImageDraw.Draw(new_mask)
            
            # Rellenar áreas expandidas en la máscara (blanco = áreas a generar)
            if expand_up > 0:
                draw.rectangle([(0, 0), (new_width, expand_up)], fill=(100, 255, 100, 255))
            if expand_down > 0:
                draw.rectangle([(0, new_height - expand_down), (new_width, new_height)], fill=(100, 255, 100, 255))
            if expand_left > 0:
                draw.rectangle([(0, expand_up), (expand_left, new_height - expand_down)], fill=(100, 255, 100, 255))
            if expand_right > 0:
                draw.rectangle([(new_width - expand_right, expand_up), (new_width, new_height - expand_down)], fill=(100, 255, 100, 255))
                
            return new_image, new_mask
            
        except Exception as e:
            print("Error in expand_canvas:", e)
            return image, mask

    # --- Interfaz de usuario ---
    def ui(self, *args, **kwargs):
        def update_ref_info(image):
            if image is None:
                return "Reference image aspect ratio: *no image*"
            else:
                return f"Reference image aspect ratio: {round(image.size[0]/image.size[1],3)} ({image.size[0]} × {image.size[1]})"
        def update_depth_info(image):
            if image is None:
                return "Depth image aspect ratio: *no image*"
            else:
                return f"Depth image aspect ratio: {round(image.size[0]/image.size[1],3)} ({image.size[0]} × {image.size[1]})"
        
        # Función que actualiza info y previsualización en Canny
        def update_canny_all(image, low, high, detect, imgres):
            info = update_ref_info(image)
            proc = self.run_canny_preprocessor(image, low, high, detect, imgres)
            return info, proc

        with InputAccordion(False, label=self.title()) as enabled:
            # Pestaña Canny
            with gradio.Tab("Canny", id="F2E_FT_Canny"):
                gradio.Markdown("Use an appropriately Flux Canny checkpoint.")
                with gradio.Row():
                    with gradio.Column():
                        canny_image = gradio.Image(label="Reference Image", type="pil", height=500, sources=["upload", "clipboard"])
                        canny_ref_info = gradio.Markdown("Reference image aspect ratio: *no image*")
                    with gradio.Column():
                        canny_preproc_preview = gradio.Image(label="Preprocessor Preview", type="pil", height=500)
                # Primera fila: Low Threshold, High Threshold y Run preprocessor en la misma línea
                with gradio.Row():
                    canny_low_threshold = gradio.Slider(label="Low Threshold", minimum=0, maximum=255, step=1, value=100)
                    canny_high_threshold = gradio.Slider(label="High Threshold", minimum=0, maximum=255, step=1, value=200)
                    run_canny_button = gradio.Button("Run preprocessor", visible=False)
                # Segunda fila: Detect Resolution e Image Resolution
                with gradio.Row():
                    canny_detect_resolution = gradio.Slider(label="Detect Resolution", minimum=128, maximum=1024, step=1, value=512)
                    canny_image_resolution = gradio.Slider(label="Image Resolution", minimum=128, maximum=1024, step=1, value=512)
                # Tercera fila: Strength y Start/End
                with gradio.Row():
                    canny_strength = gradio.Slider(label="Strength", minimum=0.0, maximum=2.0, step=0.01, value=1.0)
                    canny_time = RangeSlider(label="Start / End", minimum=0.0, maximum=1.0, step=0.01, value=(0.0, 0.8))
                
                # Actualización automática solo al soltar los controles (on release)
                canny_low_threshold.release(fn=update_canny_all, inputs=[canny_image, canny_low_threshold, canny_high_threshold, canny_detect_resolution, canny_image_resolution], outputs=[canny_ref_info, canny_preproc_preview])
                canny_high_threshold.release(fn=update_canny_all, inputs=[canny_image, canny_low_threshold, canny_high_threshold, canny_detect_resolution, canny_image_resolution], outputs=[canny_ref_info, canny_preproc_preview])
                canny_detect_resolution.release(fn=update_canny_all, inputs=[canny_image, canny_low_threshold, canny_high_threshold, canny_detect_resolution, canny_image_resolution], outputs=[canny_ref_info, canny_preproc_preview])
                canny_image_resolution.release(fn=update_canny_all, inputs=[canny_image, canny_low_threshold, canny_high_threshold, canny_detect_resolution, canny_image_resolution], outputs=[canny_ref_info, canny_preproc_preview])
                canny_image.change(fn=update_canny_all, inputs=[canny_image, canny_low_threshold, canny_high_threshold, canny_detect_resolution, canny_image_resolution], outputs=[canny_ref_info, canny_preproc_preview])
                # También se deja el botón por si se desea una acción manual
                run_canny_button.click(fn=self.run_canny_preprocessor, inputs=[canny_image, canny_low_threshold, canny_high_threshold, canny_detect_resolution, canny_image_resolution], outputs=canny_preproc_preview)
                
            # Pestaña Depth
            with gradio.Tab("Depth", id="F2E_FT_Depth"):
                gradio.Markdown("Use an appropriately Flux Depth checkpoint..")
                with gradio.Row():
                    with gradio.Column():
                        depth_image = gradio.Image(label="Reference Image", type="pil", height=500, sources=["upload", "clipboard"])
                        depth_ref_info = gradio.Markdown("Reference image aspect ratio: *no image*")
                    with gradio.Column():
                        depth_preproc_preview = gradio.Image(label="Preprocessor Preview", type="pil", height=500)
                # Se utiliza un Dropdown para seleccionar el procesador y se oculta el botón manual
                with gradio.Row():
                    depth_processor_selector = gradio.Dropdown(label="Processor", choices=["depth_zoe", "depth_midas", "depth_leres", "depth_leres++"], value="depth_zoe")
                    run_depth_button = gradio.Button("Run preprocessor", visible=False)
                with gradio.Row():
                    depth_strength = gradio.Slider(label="Strength", minimum=0.0, maximum=2.0, step=0.01, value=1.0)
                    depth_time = RangeSlider(label="Start / End", minimum=0.0, maximum=1.0, step=0.01, value=(0.0, 0.8))
                
                # Función que actualiza la información y previsualización en Depth
                def update_depth_all(image, processor):
                    info = update_ref_info(image)
                    proc = self.run_depth_preprocessor(image, processor)
                    return info, proc

                # Ejecutar automáticamente al cambiar la imagen o el procesador
                depth_image.change(fn=update_depth_all, inputs=[depth_image, depth_processor_selector], outputs=[depth_ref_info, depth_preproc_preview])
                depth_processor_selector.change(fn=update_depth_all, inputs=[depth_image, depth_processor_selector], outputs=[depth_ref_info, depth_preproc_preview])
                
            # Pestaña Fill (modificada para incluir outpainting)
            with gradio.Tab("Fill", id="F2E_FT_f"):
                gradio.Markdown("Use an appropriately Flux Fill checkpoint.")
                with gradio.Row():
                    fill_image = ForgeCanvas(
                        height=500,
                        contrast_scribbles=shared.opts.img2img_inpaint_mask_high_contrast,
                        scribble_color=shared.opts.img2img_inpaint_mask_brush_color,
                        scribble_color_fixed=True,
                        scribble_alpha=75,
                        scribble_alpha_fixed=True,
                        scribble_softness_fixed=True
                    )
                
                # Añadimos nueva sección para outpainting
                with gradio.Accordion("Outpaint Options", open=False):
                    # Sliders para controlar la expansión en cada dirección
                    with gradio.Row():
                        fill_expand_up = gradio.Slider(label="Expand Up ↑", minimum=0, maximum=512, step=64, value=0)
                        fill_expand_down = gradio.Slider(label="Expand Down ↓", minimum=0, maximum=512, step=64, value=0)
                    with gradio.Row():
                        fill_expand_left = gradio.Slider(label="Expand Left ←", minimum=0, maximum=512, step=64, value=0)
                        fill_expand_right = gradio.Slider(label="Expand Right →", minimum=0, maximum=512, step=64, value=0)
                    
                    # Botón para aplicar expansión
                    with gradio.Row():
                        apply_outpaint_button = gradio.Button("Apply Outpaint")
                    
                    # Información sobre dimensiones actuales y nuevas
                    fill_dimensions_info = gradio.Markdown("Current dimensions: N/A")
                    
                    # Función para actualizar la información de dimensiones
                    def update_dimensions_info(image, up, down, left, right):
                        if image is None:
                            return "Current dimensions: N/A"
                        width, height = image.size
                        new_width = width + left + right
                        new_height = height + up + down
                        return f"Current dimensions: {width}×{height}"
                        #return f"Current dimensions: {width}×{height} → New dimensions: {new_width}×{new_height}"
                    # Actualizar info al cambiar valores
                    fill_image.background.change(
                        fn=update_dimensions_info, 
                        inputs=[fill_image.background, fill_expand_up, fill_expand_down, fill_expand_left, fill_expand_right], 
                        outputs=fill_dimensions_info
                    )
                    
                    for slider in [fill_expand_up, fill_expand_down, fill_expand_left, fill_expand_right]:
                        slider.release(
                            fn=update_dimensions_info, 
                            inputs=[fill_image.background, fill_expand_up, fill_expand_down, fill_expand_left, fill_expand_right], 
                            outputs=fill_dimensions_info
                        )
                    
                    # Función para aplicar outpaint
                    def apply_outpaint(background, foreground, up, down, left, right):
                        if background is None:
                            return background, foreground
                        new_bg, new_fg = self.expand_canvas(background, foreground, up, down, left, right)
                        return new_bg, new_fg
                    
                    # Conectar botón a función
                    apply_outpaint_button.click(
                        fn=apply_outpaint, 
                        inputs=[fill_image.background, fill_image.foreground, 
                                fill_expand_up, fill_expand_down, fill_expand_left, fill_expand_right],
                        outputs=[fill_image.background, fill_image.foreground]
                    )
                
                with gradio.Row():
                    fill_strength = gradio.Slider(label="Strength", minimum=0.0, maximum=2.0, step=0.01, value=1.0, visible=False)
                    fill_time = RangeSlider(label=" Start / End", minimum=0.0, maximum=1.0, step=0.01, value=(0.0, 1.0), visible=False)

            # Pestaña Redux (se mantiene igual)
            with gradio.Tab("Redux", id="F2E_FT_r1"):
                gradio.Markdown("Use an appropriately Flux checkpoint.")
                with gradio.Row():
                    with gradio.Column():
                        redux_image1 = gradio.Image(show_label=False, type="pil", height=500, sources=["upload", "clipboard"])
                    with gradio.Column():
                        redux_str1 = gradio.Slider(label="Strength", minimum=0.0, maximum=2.0, step=0.01, value=1.0)
                        redux_time1 = RangeSlider(label="Start / End", minimum=0.0, maximum=1.0, step=0.01, value=(0.0, 0.8))
                        swap12 = gradio.Button("swap redux 1 and 2")
                        swap13 = gradio.Button("swap redux 1 and 3")
                        swap14 = gradio.Button("swap redux 1 and 4")
            with gradio.Tab("Redux-2", id="F2E_FT_r2"):
                gradio.Markdown("Use an appropriately Flux checkpoint.")
                with gradio.Row():
                    with gradio.Column():
                        redux_image2 = gradio.Image(show_label=False, type="pil", height=500, sources=["upload", "clipboard"])
                    with gradio.Column():
                        redux_str2 = gradio.Slider(label="Strength", minimum=0.0, maximum=2.0, step=0.01, value=1.0)
                        redux_time2 = RangeSlider(label="Start / End", minimum=0.0, maximum=1.0, step=0.01, value=(0.0, 0.8))
                        swap21 = gradio.Button("swap redux 2 and 1")
                        swap23 = gradio.Button("swap redux 2 and 3")
                        swap24 = gradio.Button("swap redux 2 and 4")
            with gradio.Tab("Redux-3", id="F2E_FT_r3"):
                gradio.Markdown("Use an appropriately Flux checkpoint.")
                with gradio.Row():
                    with gradio.Column():
                        redux_image3 = gradio.Image(show_label=False, type="pil", height=500, sources=["upload", "clipboard"])
                    with gradio.Column():
                        redux_str3 = gradio.Slider(label="Strength", minimum=0.0, maximum=2.0, step=0.01, value=1.0)
                        redux_time3 = RangeSlider(label="Start / End", minimum=0.0, maximum=1.0, step=0.01, value=(0.0, 0.8))
                        swap31 = gradio.Button("swap redux 3 and 1")
                        swap32 = gradio.Button("swap redux 3 and 2")
                        swap34 = gradio.Button("swap redux 3 and 4")
            with gradio.Tab("Redux-4", id="F2E_FT_r4"):
                gradio.Markdown("Use an appropriately Flux checkpoint.")
                with gradio.Row():
                    with gradio.Column():
                        redux_image4 = gradio.Image(show_label=False, type="pil", height=500, sources=["upload", "clipboard"])
                    with gradio.Column():
                        redux_str4 = gradio.Slider(label="Strength", minimum=0.0, maximum=2.0, step=0.01, value=1.0)
                        redux_time4 = RangeSlider(label="Start / End", minimum=0.0, maximum=1.0, step=0.01, value=(0.0, 0.8))
                        swap41 = gradio.Button("swap redux 4 and 1")
                        swap42 = gradio.Button("swap redux 4 and 2")
                        swap43 = gradio.Button("swap redux 4 and 3")
            def redux_swap(image1, image2, str1, str2, time1, time2):
                return image2, image1, str2, str1, time2, time1
            swap12.click(redux_swap, inputs=[redux_image1, redux_image2, redux_str1, redux_str2, redux_time1, redux_time2],
                         outputs=[redux_image1, redux_image2, redux_str1, redux_str2, redux_time1, redux_time2])
            swap13.click(redux_swap, inputs=[redux_image1, redux_image3, redux_str1, redux_str3, redux_time1, redux_time3],
                         outputs=[redux_image1, redux_image3, redux_str1, redux_str3, redux_time1, redux_time3])
            swap14.click(redux_swap, inputs=[redux_image1, redux_image4, redux_str1, redux_str4, redux_time1, redux_time4],
                         outputs=[redux_image1, redux_image4, redux_str1, redux_str4, redux_time1, redux_time4])
            swap21.click(redux_swap, inputs=[redux_image2, redux_image1, redux_str2, redux_str1, redux_time2, redux_time1],
                         outputs=[redux_image2, redux_image1, redux_str2, redux_str1, redux_time2, redux_time1])
            swap23.click(redux_swap, inputs=[redux_image2, redux_image3, redux_str2, redux_str3, redux_time2, redux_time3],
                         outputs=[redux_image2, redux_image3, redux_str2, redux_str3, redux_time2, redux_time3])
            swap24.click(redux_swap, inputs=[redux_image2, redux_image4, redux_str2, redux_str4, redux_time2, redux_time4],
                         outputs=[redux_image2, redux_image4, redux_str2, redux_str4, redux_time2, redux_time4])
            swap31.click(redux_swap, inputs=[redux_image3, redux_image1, redux_str3, redux_str1, redux_time3, redux_time1],
                         outputs=[redux_image3, redux_image1, redux_str3, redux_str1, redux_time3, redux_time1])
            swap32.click(redux_swap, inputs=[redux_image3, redux_image2, redux_str3, redux_str2, redux_time3, redux_time2],
                         outputs=[redux_image3, redux_image2, redux_str3, redux_str2, redux_time3, redux_time2])
            swap34.click(redux_swap, inputs=[redux_image3, redux_image4, redux_str3, redux_str4, redux_time3, redux_time4],
                         outputs=[redux_image3, redux_image4, redux_str3, redux_str4, redux_time3, redux_time4])
            swap41.click(redux_swap, inputs=[redux_image4, redux_image1, redux_str4, redux_str1, redux_time4, redux_time1],
                         outputs=[redux_image4, redux_image1, redux_str4, redux_str1, redux_time4, redux_time1])
            swap42.click(redux_swap, inputs=[redux_image4, redux_image2, redux_str4, redux_str2, redux_time4, redux_time2],
                         outputs=[redux_image4, redux_image2, redux_str4, redux_str2, redux_time4, redux_time2])
            swap43.click(redux_swap, inputs=[redux_image4, redux_image3, redux_str4, redux_str3, redux_time4, redux_time3],
                         outputs=[redux_image4, redux_image3, redux_str4, redux_str3, redux_time4, redux_time3])
            with gradio.Accordion('Text encoders control', open=False, visible=False):
                te_device = gradio.Radio(label="Device for text encoders", choices=["default", "cpu", "gpu", "gpu-2"], value="default")
                with gradio.Row():
                    flux_use_T5 = gradio.Checkbox(value=flux1tools.flux_use_T5, label="Flux: Use T5")
                    flux_use_CL = gradio.Checkbox(value=flux1tools.flux_use_CL, label="Flux: Use CLIP (pooled)")
        self.infotext_fields = [
            (enabled, lambda d: d.get("fmp_enabled", False)),
            (te_device, "fmp_te_device"),
            (flux_use_T5, "fmp_fluxT5"),
            (flux_use_CL, "fmp_fluxCL"),
        ]
        def clearCondCache():
            flux1tools.clearConds = True
        enabled.change(fn=clearCondCache, inputs=None, outputs=None)
        flux_use_T5.change(fn=clearCondCache, inputs=None, outputs=None)
        flux_use_CL.change(fn=clearCondCache, inputs=None, outputs=None)
        
        # Añadir nuevos componentes de outpainting a la lista de retorno
        return enabled, te_device, flux_use_T5, flux_use_CL, \
               canny_image, canny_ref_info, canny_preproc_preview, canny_low_threshold, canny_high_threshold, canny_detect_resolution, canny_image_resolution, canny_strength, canny_time, \
               depth_image, depth_ref_info, depth_preproc_preview, depth_processor_selector, depth_strength, depth_time, \
               redux_image1, redux_image2, redux_image3, redux_image4, \
               redux_str1, redux_str2, redux_str3, redux_str4, \
               redux_time1, redux_time2, redux_time3, redux_time4, \
               fill_image.background, fill_image.foreground, fill_strength, fill_time, \
               fill_expand_up, fill_expand_down, fill_expand_left, fill_expand_right, fill_dimensions_info

    def after_extra_networks_activate(self, p, *script_args, **kwargs):
        enabled = script_args[0]
        if enabled:
            te_device = script_args[1]
            match te_device:
                case "gpu-2":
                    memory_management.text_encoder_device = flux1tools.patched_text_encoder_gpu2
                case "gpu":
                    memory_management.text_encoder_device = flux1tools.patched_text_encoder_gpu
                case "cpu":
                    memory_management.text_encoder_device = flux1tools.patched_text_encoder_cpu
                case _:
                    pass

    def process(self, params, *script_args, **kwargs):
        # Desempaquetamos los argumentos incluyendo los nuevos componentes de outpainting
        (enabled, te_device, flux_use_T5, flux_use_CL,
         canny_image, canny_ref_info, canny_preproc_preview, canny_low_threshold, canny_high_threshold, canny_detect_resolution, canny_image_resolution, canny_strength, canny_time,
         depth_image, depth_ref_info, depth_preproc_preview, depth_processor_selector, depth_strength, depth_time,
         redux_image1, redux_image2, redux_image3, redux_image4,
         redux_str1, redux_str2, redux_str3, redux_str4,
         redux_time1, redux_time2, redux_time3, redux_time4,
         fill_image, fill_mask, fill_strength, fill_time,
         fill_expand_up, fill_expand_down, fill_expand_left, fill_expand_right, fill_dimensions_info) = script_args

    def process_before_every_sampling(self, params, *script_args, **kwargs):
        (enabled, te_device, flux_use_T5, flux_use_CL,
         canny_image, canny_ref_info, canny_preproc_preview, canny_low_threshold, canny_high_threshold, canny_detect_resolution, canny_image_resolution, canny_strength, canny_time,
         depth_image, depth_ref_info, depth_preproc_preview, depth_processor_selector, depth_strength, depth_time,
         redux_image1, redux_image2, redux_image3, redux_image4,
         redux_str1, redux_str2, redux_str3, redux_str4,
         redux_time1, redux_time2, redux_time3, redux_time4,
         fill_image, fill_mask, fill_strength, fill_time,
         fill_expand_up, fill_expand_down, fill_expand_left, fill_expand_right, fill_dimensions_info) = script_args
         
        if enabled:
            if not params.sd_model.is_webui_legacy_model():
                x = kwargs['x']
                n, c, h, w = x.size()
                # Prioridad: Fill > Canny > Depth
                if fill_image is not None and fill_mask is not None:
                    mask_A = fill_mask.getchannel('A').convert('L')
                    mask_A_I = mask_A.point(lambda v: 0 if v > 128 else 255)
                    mask_A = mask_A.point(lambda v: 255 if v > 128 else 0)
                    mask = Image.merge('RGBA', (mask_A_I, mask_A_I, mask_A_I, mask_A))
                    image = Image.alpha_composite(fill_image, mask).convert('RGB')
                    image = image.resize((w * 8, h * 8))
                    image_np = np.array(image) / 255.0
                    image_np = np.transpose(image_np, (2, 0, 1))
                    image_tensor = torch.tensor(image_np).unsqueeze(0)
                    latent = images_tensor_to_samples(image_tensor, approximation_indexes.get(shared.opts.sd_vae_encode_method), params.sd_model)
                    
                    mask_resized = mask_A.resize((w * 8, h * 8))
                    mask_np = np.array(mask_resized) / 255.0
                    mask_tensor = torch.tensor(mask_np).unsqueeze(0).unsqueeze(0)
                    mask_tensor = mask_tensor[:, 0, :, :]
                    mask_tensor = mask_tensor.view(1, h, 8, w, 8)
                    mask_tensor = mask_tensor.permute(0, 2, 4, 1, 3)
                    mask_tensor = mask_tensor.reshape(1, 64, h, w)
                    
                    flux1tools.latent = torch.cat([latent, mask_tensor.to(latent.device)], dim=1)
                    flux1tools.unmasked_latent = None
                    del image_tensor, mask_tensor
                    flux1tools.start = 0.0
                    flux1tools.end = 1.0
                    flux1tools.strength = 1.0
                elif canny_preproc_preview is not None and canny_strength > 0:
                    image = canny_preproc_preview.resize((w * 8, h * 8))
                    image = np.array(image) / 255.0
                    image = np.transpose(image, (2, 0, 1))
                    image = torch.tensor(image).unsqueeze(0)
                    latent = images_tensor_to_samples(image, approximation_indexes.get(shared.opts.sd_vae_encode_method), params.sd_model)
                    flux1tools.latent = latent
                    flux1tools.unmasked_latent = None
                    del image
                    flux1tools.start = canny_time[0]
                    flux1tools.end = canny_time[1]
                    flux1tools.strength = canny_strength
                elif depth_preproc_preview is not None and depth_strength > 0:
                    image = depth_preproc_preview.resize((w * 8, h * 8))
                    image = np.array(image) / 255.0
                    image = np.transpose(image, (2, 0, 1))
                    image = torch.tensor(image).unsqueeze(0)
                    latent = images_tensor_to_samples(image, approximation_indexes.get(shared.opts.sd_vae_encode_method), params.sd_model)
                    flux1tools.latent = latent
                    flux1tools.unmasked_latent = None
                    del image
                    flux1tools.start = depth_time[0]
                    flux1tools.end = depth_time[1]
                    flux1tools.strength = depth_strength
                else:
                    flux1tools.latent = None

                redux_images = [redux_image1, redux_image2, redux_image3, redux_image4]
                redux_strengths = [redux_str1, redux_str2, redux_str3, redux_str4]
                redux_times = [redux_time1, redux_time2, redux_time3, redux_time4]
                if redux_images != [None, None, None, None] and redux_strengths != [0, 0, 0, 0]:
                    from transformers import SiglipImageProcessor, SiglipVisionModel
                    from diffusers.pipelines.flux.modeling_flux import ReduxImageEncoder
                    embeds = []
                    for i in range(len(redux_images)):
                        if redux_images[i] is None or redux_strengths[i] == 0:
                            continue
                        feature = SiglipImageProcessor.from_pretrained("Runware/FLUX.1-Redux-dev", subfolder="feature_extractor")
                        image = feature.preprocess(images=redux_images[i], do_resize=True, return_tensors="pt", do_convert_rgb=True)
                        del feature
                        encoder = SiglipVisionModel.from_pretrained("Runware/FLUX.1-Redux-dev", subfolder="image_encoder")
                        image_enc_hidden_states = encoder(**image).last_hidden_state
                        del encoder
                        embedder = ReduxImageEncoder.from_pretrained("Runware/FLUX.1-Redux-dev", subfolder="image_embedder")
                        embeds.append((redux_strengths[i] * embedder(image_enc_hidden_states).image_embeds, redux_times[i][0], redux_times[i][1]))
                        del embedder, image_enc_hidden_states
                    flux1tools.image_embeds = embeds
                else:
                    flux1tools.image_embeds = None

                def apply_control(self):
                    lastStep = self.total_sampling_steps - 1
                    thisStep = self.sampling_step
                    if flux1tools.image_embeds is not None:
                        embeds = flux1tools.image_embeds
                        cond = self.text_cond["crossattn"]
                        for e in embeds:
                            if thisStep >= e[1] * lastStep and thisStep <= e[2] * lastStep:
                                image_embeds = e[0].repeat_interleave(len(self.text_cond["crossattn"]), dim=0)
                                image_embeds *= (256 / 729)
                                cond = torch.cat([cond, image_embeds.to(cond.device)], dim=1)
                                del image_embeds
                        cond = torch.sum(cond, dim=0, keepdim=True)
                        self.text_cond["crossattn"] = cond
                    if flux1tools.latent is not None:
                        if thisStep >= flux1tools.start * lastStep and thisStep <= flux1tools.end * lastStep:
                            latent_strength = flux1tools.latent * flux1tools.strength
                            shared.sd_model.forge_objects.unet.extra_concat_condition = latent_strength
                        else:
                            if flux1tools.unmasked_latent is not None:
                                shared.sd_model.forge_objects.unet.extra_concat_condition = flux1tools.unmasked_latent
                            else:
                                shared.sd_model.forge_objects.unet.extra_concat_condition = flux1tools.latent * 0.0
                on_cfg_denoiser(apply_control)
        return

    def postprocess(self, params, processed, *args):
        enabled = args[0]
        if enabled:
            Flux.get_learned_conditioning = flux1tools.glc_backup_flux
            memory_management.text_encoder_device = flux1tools.text_encoder_device_backup
            if flux1tools.sigmasBackup is not None:
                shared.sd_model.forge_objects.unet.model.predictor.sigmas = flux1tools.sigmasBackup
                flux1tools.sigmasBackup = None
            shared.sd_model.forge_objects.unet.extra_concat_condition = None
            flux1tools.image_embeds = None
            flux1tools.latent = None
            flux1tools.unmasked_latent = None
            remove_current_script_callbacks()
        return
