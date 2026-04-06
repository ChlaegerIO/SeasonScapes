from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
import json

from paint_engine import utils
from eval_engine.eval_utils import ssim, psnr, mse, calculate_fid
from eval_engine.lpipsPyTorch import lpips
from data_engine.utils.general import getHomePath
from paint_engine.paint_pipeline import Paint_pipeline



class Eval_pipeline():
    def __init__(self, eval_path, paint_cfg, trigger_word="inpaint"):
        self.home_path = getHomePath()
        self.eval_path = Path(self.home_path, eval_path)
        self.paint_cfg = paint_cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.paint = Paint_pipeline(paint_cfg)
        self.trigger_word = trigger_word
        if not self.eval_path.exists():
            self.eval_path.mkdir(parents=True, exist_ok=True)

    def evaluate(self):
        print("\nEvaluating pipeline...")
        # Render data
        print("Rendering data...")
        self.paint.evaluation()

        # copy and preprocess data
        infer_path = Path(self.paint_cfg.log.exp_path, "eval_metrics")
        targetTransforms_path = Path(self.paint_cfg.dataset.fileTransformMatrix)
        
        # Load transform file
        self.transform_path = Path(self.home_path, targetTransforms_path)
        with open(self.transform_path, 'r') as f:
            transforms = json.load(f)
        
        first_key = next(iter(transforms))
        transform = transforms[first_key]
        target_path = Path(self.home_path.parent, transform["file_path"]).parent

        saveTarget_path = Path(self.eval_path, "target")
        saveInfer_path = Path(self.eval_path, "inference")
        output_file = Path(self.eval_path, "metrics.md")
        if not saveTarget_path.exists():
            saveTarget_path.mkdir(parents=True, exist_ok=True)
        if not saveInfer_path.exists():
            saveInfer_path.mkdir(parents=True, exist_ok=True)            

        name_list = []
        psnr_list = []
        ssim_list = []
        mse_list = []
        lpips_list = []
        fid_list = []
        for img_path in tqdm(list(infer_path.glob("*.png"))):
            img_name = img_path.name
            if self.trigger_word in img_name:
                img_id = "_".join(img_name.split("_")[:3])
                eval_mask_path = Path(img_path.parent, f"{img_id}_eval_mask.png")
                target_file = Path(target_path, f"{img_id}.jpg")
                
                inference_img = Image.open(img_path)
                eval_mask = Image.open(eval_mask_path)
                target_img = Image.open(target_file)
                if inference_img.size != target_img.size:
                    print(f"Match img sizes: {inference_img.size} != {target_img.size}")
                    target_img = target_img.resize(inference_img.size, Image.BILINEAR)

                # Convert to float32 and normalize to [0, 1]
                inference_img = utils.pil2tensor(inference_img, 'cpu')
                target_img = utils.pil2tensor(target_img, 'cpu')
                eval_mask = utils.pil2tensor(eval_mask, 'cpu')
            
                masked_inference = inference_img * eval_mask
                masked_target = target_img * eval_mask

                # calculate metrics
                psnr_value = psnr(masked_target[0], masked_inference[0]).mean().item()
                ssim_value = ssim(masked_target, masked_inference).item()
                mse_value = mse(masked_target[0], masked_inference[0]).mean().item()
                lpips_value = lpips(masked_target, masked_inference).detach().cpu().numpy().mean()
                fid_value = calculate_fid(masked_target, masked_inference, False, 1)

                name_list.append(img_name)
                psnr_list.append(psnr_value)
                ssim_list.append(ssim_value)
                mse_list.append(mse_value)
                lpips_list.append(lpips_value)
                fid_list.append(fid_value)

                utils.save_tensor_image(masked_inference, Path(saveInfer_path, "masked", f"{img_id}_masked.png").as_posix())
                utils.save_tensor_image(masked_target, Path(saveTarget_path, "masked", f"{img_id}_masked.png").as_posix())

        mean_psnr = np.mean(psnr_list)
        mean_ssim = np.mean(ssim_list)
        mean_mse = np.mean(mse_list)
        mean_lpips = np.mean(lpips_list)
        mean_fid = np.mean(fid_list)

        std_psnr = np.std(psnr_list)
        std_ssim = np.std(ssim_list)
        std_mse = np.std(mse_list)
        std_lpips = np.std(lpips_list)
        std_fid = np.std(fid_list)

        median_psnr = np.median(psnr_list)
        median_ssim = np.median(ssim_list)
        median_mse = np.median(mse_list)
        median_lpips = np.median(lpips_list)
        median_fid = np.median(fid_list)

        print(f"Mean PSNR: {mean_psnr:.2f}, Mean SSIM: {mean_ssim:.3f}, Mean MSE: {mean_mse:.4f}, Mean LPIPS: {mean_lpips:.3f}, Mean FID: {mean_fid:.3f}")
        print(f"Median PSNR: {median_psnr:.2f}, Median SSIM: {median_ssim:.3f}, Median MSE: {median_mse:.4f}, Median LPIPS: {median_lpips:.3f}, Median FID: {median_fid:.3f}")
        print(f"Std PSNR: {std_psnr:.2f}, Std SSIM: {std_ssim:.3f}, Std MSE: {std_mse:.4f}, Std LPIPS: {std_lpips:.3f}, Std FID: {std_fid:.3f}")

        # Write metrics to Markdown file
        print("Writing metrics to file to ", Path(output_file, "metrics.md"))
        with open(output_file, "w") as f:
            # Write LaTeX table header
            f.write("| Image Name | PSNR | SSIM | MSE | LPIPS | FID |\n")
            f.write("|------------|------|------|-----|-------|-----|\n")
            for i in range(len(name_list)):
                f.write(f"| {name_list[i]} | {psnr_list[i]:.2f} | {ssim_list[i]:.3f} | {mse_list[i]:.4f} | {lpips_list[i]:.3f} | {fid_list[i]:.3f} |\n")
                print(f"Metrics for {name_list[i]}:")
                print(f"PSNR: {psnr_list[i]:.2f}, SSIM: {ssim_list[i]:.3f}, MSE: {mse_list[i]:.4f}, LPIPS: {lpips_list[i]:.3f}, FID: {fid_list[i]:.3f}")
            f.write("|------------|------|------|-----|-------|-----|\n")
            f.write(f"| **Mean** | {mean_psnr:.2f} | {mean_ssim:.3f} | {mean_mse:.4f} | {mean_lpips:.3f} | {mean_fid:.3f} |\n")
            f.write(f"| **Std** | {std_psnr:.2f} | {std_ssim:.3f} | {std_mse:.4f} | {std_lpips:.3f} | {std_fid:.3f} |\n")
            f.write(f"| **Median** | {median_psnr:.2f} | {median_ssim:.3f} | {median_mse:.4f} | {median_lpips:.3f} | {median_fid:.3f} |\n")

    def evaluate_folder(self, with_eval_mask=False):
        print("\nEvaluating folder...")
        eval_path = Path(self.eval_path, "inference")
        target_path = Path(self.eval_path, "target")

        name_list = []
        psnr_list = []
        ssim_list = []
        mse_list = []
        lpips_list = []
        fid_list = []
        for img in tqdm(list(eval_path.glob("*.png"))):
            img_name = img.name
            target_file = Path(target_path, img_name)
            if not target_file.exists():
                print(f"Target image {target_file} does not exist.")
                continue
                            
            inference_img = Image.open(img)
            target_img = Image.open(target_file.as_posix())
            if inference_img.size != target_img.size:
                print(f"\nMatch img sizes: {inference_img.size} != {target_img.size}")
                target_img = target_img.resize(inference_img.size, Image.BILINEAR)

            # Convert to float32 and normalize to [0, 1]
            inference_img = utils.pil2tensor(inference_img, 'cpu')
            target_img = utils.pil2tensor(target_img, 'cpu')

            # calculate eval mask
            masked_inference = inference_img
            masked_target = target_img
            if with_eval_mask:
                eval_mask_path = Path(eval_path, f"{img_name.split('.')[0]}_eval_mask.png")
                eval_mask = Image.open(eval_mask_path)
                eval_mask = utils.pil2tensor(eval_mask, 'cpu')
                masked_inference = inference_img * eval_mask
                masked_target = target_img * eval_mask
        
            # calculate metrics
            psnr_value = psnr(masked_target[0], masked_inference[0]).mean().item()
            ssim_value = ssim(masked_target, masked_inference).item()
            mse_value = mse(masked_target[0], masked_inference[0]).mean().item()
            lpips_value = lpips(masked_target, masked_inference).detach().cpu().numpy().mean()
            fid_value = calculate_fid(masked_target, masked_inference, False, 1)  # Assuming you have a function to calculate FID

            name_list.append(img_name)
            psnr_list.append(psnr_value)
            ssim_list.append(ssim_value)
            mse_list.append(mse_value)
            lpips_list.append(lpips_value)
            fid_list.append(fid_value)

            utils.save_tensor_image(masked_inference, Path(eval_path, "masked", f"{img_name.split('.')[0]}_masked.png").as_posix())
            utils.save_tensor_image(masked_target, Path(target_path, "masked", f"{img_name.split('.')[0]}_masked.png").as_posix())

        # clean nan, inf, -inf
        psnr_array = np.array(psnr_list)
        ssim_array = np.array(ssim_list)
        mse_array = np.array(mse_list)
        lpips_array = np.array(lpips_list)
        fid_array = np.array(fid_list)
        cleaned_psnr = psnr_array[~np.isinf(psnr_array) & ~np.isnan(psnr_array)]
        cleaned_ssim = ssim_array[~np.isinf(ssim_array) & ~np.isnan(ssim_array)]
        cleaned_mse = mse_array[~np.isinf(mse_array) & ~np.isnan(mse_array)]
        cleaned_lpips = lpips_array[~np.isinf(lpips_array) & ~np.isnan(lpips_array)]
        cleaned_fid = fid_array[~np.isinf(fid_array) & ~np.isnan(fid_array)]
        
        mean_psnr = np.mean(cleaned_psnr)
        mean_ssim = np.mean(cleaned_ssim)
        mean_mse = np.mean(cleaned_mse)
        mean_lpips = np.mean(cleaned_lpips)
        mean_fid = np.mean(cleaned_fid)

        std_psnr = np.std(cleaned_psnr)
        std_ssim = np.std(cleaned_ssim)
        std_mse = np.std(cleaned_mse)
        std_lpips = np.std(cleaned_lpips)
        std_fid = np.std(cleaned_fid)

        median_psnr = np.median(cleaned_psnr)
        median_ssim = np.median(cleaned_ssim)
        median_mse = np.median(cleaned_mse)
        median_lpips = np.median(cleaned_lpips)
        median_fid = np.median(cleaned_fid)

        print(f"Mean PSNR: {mean_psnr:.2f}, Mean SSIM: {mean_ssim:.3f}, Mean MSE: {mean_mse:.4f}, Mean LPIPS: {mean_lpips:.3f}, Mean FID: {mean_fid:.3f}")
        print(f"Median PSNR: {median_psnr:.2f}, Median SSIM: {median_ssim:.3f}, Median MSE: {median_mse:.4f}, Median LPIPS: {median_lpips:.3f}, Median FID: {median_fid:.3f}")
        print(f"Std PSNR: {std_psnr:.2f}, Std SSIM: {std_ssim:.3f}, Std MSE: {std_mse:.4f}, Std LPIPS: {std_lpips:.3f}, Std FID: {std_fid:.3f}")

        # Write metrics to Markdown file
        print("Writing metrics to file to ", Path(self.eval_path, "metrics.md"))
        with open(Path(self.eval_path, "metrics.md"), "w") as f:
            # Write LaTeX table header
            f.write("| Image Name | PSNR | SSIM | MSE | LPIPS | FID |\n")
            f.write("|------------|------|------|-----|-------|-----|\n")
            for i in range(len(name_list)):
                f.write(f"| {name_list[i]} | {psnr_list[i]:.2f} | {ssim_list[i]:.3f} | {mse_list[i]:.4f} | {lpips_list[i]:.3f} | {fid_list[i]:.3f} |\n")
            f.write("|------------|------|------|-----|-------|-----|\n")
            f.write(f"| **Mean** | {mean_psnr:.2f} | {mean_ssim:.3f} | {mean_mse:.4f} | {mean_lpips:.3f} | {mean_fid:.3f} |\n")
            f.write(f"| **Std** | {std_psnr:.2f} | {std_ssim:.3f} | {std_mse:.4f} | {std_lpips:.3f} | {std_fid:.3f} |\n")
            f.write(f"| **Median** | {median_psnr:.2f} | {median_ssim:.3f} | {median_mse:.4f} | {median_lpips:.3f} | {median_fid:.3f} |\n")

    def evaluate_inpaint_folder(self, with_eval_mask=False):
        print("\nEvaluating inpaint folder...")
        eval_path = Path(self.eval_path, "inference")
        target_path = Path(self.eval_path, "target")

        name_list = []
        psnr_list = []
        ssim_list = []
        mse_list = []
        lpips_list = []
        fid_list = []
        for img in tqdm(list(eval_path.glob("*.png"))):
            img_name = img.name
            if self.trigger_word in img_name:
                img_id = "_".join(img_name.split("_")[:3])
                print(f"\nimg_id: {img_id}")
                target_file = Path(target_path, f"{img_id}.jpg")
                if not target_file.exists():
                    print(f"Target image {target_file} does not exist.")
                    continue
                                
                inference_img = Image.open(img)
                target_img = Image.open(target_file.as_posix())
                if inference_img.size != target_img.size:
                    print(f"\nMatch img sizes: {inference_img.size} != {target_img.size}")
                    target_img = target_img.resize(inference_img.size, Image.BILINEAR)

                # Convert to float32 and normalize to [0, 1]
                inference_img = utils.pil2tensor(inference_img, 'cpu')
                target_img = utils.pil2tensor(target_img, 'cpu')

                # calculate eval mask
                masked_inference = inference_img
                masked_target = target_img
                if with_eval_mask:
                    eval_mask_path = Path(eval_path, f"{img_id}_eval_mask.png")
                    eval_mask = Image.open(eval_mask_path)
                    eval_mask = utils.pil2tensor(eval_mask, 'cpu')
                    masked_inference = inference_img * eval_mask
                    masked_target = target_img * eval_mask
            
                # calculate metrics
                psnr_value = psnr(masked_target[0], masked_inference[0]).mean().item()
                ssim_value = ssim(masked_target, masked_inference).item()
                mse_value = mse(masked_target[0], masked_inference[0]).mean().item()
                lpips_value = lpips(masked_target, masked_inference).detach().cpu().numpy().mean()
                utils.save_tensor_image(masked_inference, Path(eval_path, "masked", f"{img_name.split('.')[0]}_masked.png").as_posix())
                utils.save_tensor_image(masked_target, Path(target_path, "masked", f"{img_name.split('.')[0]}_masked.png").as_posix())
                masked_target = masked_target.permute(0, 2, 3, 1).numpy()
                masked_inference = masked_inference.permute(0, 2, 3, 1).numpy()
                fid_value = calculate_fid(masked_target, masked_inference, False, 1)    # does not make sense per image, it is distribution to distribution metric

                name_list.append(img_name)
                psnr_list.append(psnr_value)
                ssim_list.append(ssim_value)
                mse_list.append(mse_value)
                lpips_list.append(lpips_value)
                fid_list.append(fid_value)

        # clean nan, inf, -inf
        psnr_array = np.array(psnr_list)
        ssim_array = np.array(ssim_list)
        mse_array = np.array(mse_list)
        lpips_array = np.array(lpips_list)
        fid_array = np.array(fid_list)
        cleaned_psnr = psnr_array[~np.isinf(psnr_array) & ~np.isnan(psnr_array)]
        cleaned_ssim = ssim_array[~np.isinf(ssim_array) & ~np.isnan(ssim_array)]
        cleaned_mse = mse_array[~np.isinf(mse_array) & ~np.isnan(mse_array)]
        cleaned_lpips = lpips_array[~np.isinf(lpips_array) & ~np.isnan(lpips_array)]
        cleaned_fid = fid_array[~np.isinf(fid_array) & ~np.isnan(fid_array)]
        
        mean_psnr = np.mean(cleaned_psnr)
        mean_ssim = np.mean(cleaned_ssim)
        mean_mse = np.mean(cleaned_mse)
        mean_lpips = np.mean(cleaned_lpips)
        mean_fid = np.mean(cleaned_fid)

        std_psnr = np.std(cleaned_psnr)
        std_ssim = np.std(cleaned_ssim)
        std_mse = np.std(cleaned_mse)
        std_lpips = np.std(cleaned_lpips)
        std_fid = np.std(cleaned_fid)

        median_psnr = np.median(cleaned_psnr)
        median_ssim = np.median(cleaned_ssim)
        median_mse = np.median(cleaned_mse)
        median_lpips = np.median(cleaned_lpips)
        median_fid = np.median(cleaned_fid)

        print(f"Mean PSNR: {mean_psnr:.2f}, Mean SSIM: {mean_ssim:.3f}, Mean MSE: {mean_mse:.4f}, Mean LPIPS: {mean_lpips:.3f}, Mean FID: {mean_fid:.3f}")
        print(f"Median PSNR: {median_psnr:.2f}, Median SSIM: {median_ssim:.3f}, Median MSE: {median_mse:.4f}, Median LPIPS: {median_lpips:.3f}, Median FID: {median_fid:.3f}")
        print(f"Std PSNR: {std_psnr:.2f}, Std SSIM: {std_ssim:.3f}, Std MSE: {std_mse:.4f}, Std LPIPS: {std_lpips:.3f}, Std FID: {std_fid:.3f}")
        
        # Write metrics to Markdown file
        print("Writing metrics to file to ", Path(self.eval_path, "metrics.md"))
        with open(Path(self.eval_path, "metrics.md"), "w") as f:
            # Write LaTeX table header
            f.write("| Image Name | PSNR | SSIM | MSE | LPIPS | FID |\n")
            f.write("|------------|------|------|-----|-------|-----|\n")
            for i in range(len(name_list)):
                f.write(f"| {name_list[i]} | {psnr_list[i]:.2f} | {ssim_list[i]:.3f} | {mse_list[i]:.4f} | {lpips_list[i]:.3f} | {fid_list[i]:.3f} |\n")
            f.write("|------------|------|------|-----|-------|-----|\n")
            f.write(f"| **Mean** | {mean_psnr:.2f} | {mean_ssim:.3f} | {mean_mse:.4f} | {mean_lpips:.3f} | {mean_fid:.3f} |\n")
            f.write(f"| **Std** | {std_psnr:.2f} | {std_ssim:.3f} | {std_mse:.4f} | {std_lpips:.3f} | {std_fid:.3f} |\n")
            f.write(f"| **Median** | {median_psnr:.2f} | {median_ssim:.3f} | {median_mse:.4f} | {median_lpips:.3f} | {median_fid:.3f} |\n")

