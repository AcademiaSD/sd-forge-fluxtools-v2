# sd-forge-fluxtools-v2

This is an implementation to use Canny, Depth, Redux and Fill Flux1.dev ControlNet Forge WebUI Extension
FluxTool code taken from DenOfEquity's forgeFlux_dualPrompt extension https://github.com/DenOfEquity/forgeFlux_dualPrompt
I have added the preprocessor system and the outpaint system, in any case, the main credit is from DenOfEquity https://github.com/DenOfEquity

![image](https://github.com/AcademiaSD/sd-forge-fluxtools-v2/blob/main/assets/fluxtoolsv2_canny.webp)


## Install
Go to the Extensions tab > Install from URL > URL for this repository.

Video instructions: https://www.youtube.com/watch?v=MHYSFBkF36s


## Requierements
> [!WARNING]  
> Easiest way to ensure necessary diffusers release is installed is to edit requirements_versions.txt in the webUI directory.
> 
> diffusers>=0.32.0
>
> controlnet-aux>=0.0.9
>
> Redux models will be downloaded on first use, just under 1Gb.
> Depth models will be downloaded on first use, just under 2.6Gb.


## Downloads
> [!NOTE]  
> Download checkpoints and move to folder models/stable-diffusion
>
> - For Redux
>   (https://huggingface.co/Academia-SD/flux1-Dev-FP8/tree/main)
>
> - For Canny 
>   (https://huggingface.co/Academia-SD/flux1-Canny-Dev-FP8/tree/main)
>
> - For Depth
>   (https://huggingface.co/Academia-SD/flux1-Depth-Dev-FP8/tree/main)
>
> - For Fill
>   (https://huggingface.co/Academia-SD/flux1-Fill-Dev-FP8/tree/main)
>

![image](https://github.com/AcademiaSD/sd-forge-fluxtools-v2/blob/main/assets/fluxtoolsv2_depth.webp)

![image](https://github.com/AcademiaSD/sd-forge-fluxtools-v2/blob/main/assets/fluxtoolsv2_fill_outpaint.png)

![image](https://github.com/AcademiaSD/sd-forge-fluxtools-v2/blob/main/assets/fluxtoolsv2_redux_simple.webp)

![image](https://github.com/AcademiaSD/sd-forge-fluxtools-v2/blob/main/assets/fluxtoolsv2_redux_multi.webp)
