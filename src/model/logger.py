import numpy as np
import os


class Logger():
    def __init__(self, config, wandb_run, eval_dir):
        self.psnr_list = list()
        self.rmse_list = list()
        self.l1_list = list()
        self.ssim_list = list()
        self.lpips_list = list()

        self.tracking_obj_iter_time_sum = 0
        self.tracking_obj_iter_time_count = 0
        self.tracking_obj_frame_time_sum = 0
        self.tracking_obj_frame_time_count = 0
        self.tracking_cam_iter_time_sum = 0
        self.tracking_cam_iter_time_count = 0
        self.tracking_cam_frame_time_sum = 0
        self.tracking_cam_frame_time_count = 0

        self.config = config
        self.wandb_run = wandb_run
        self.eval_dir = eval_dir

    def log_time_stats(self):
        self.tracking_obj_iter_time_count = max(self.tracking_obj_iter_time_count, 1)
        self.tracking_cam_iter_time_count = max(self.tracking_cam_iter_time_count, 1)
        self.tracking_obj_frame_time_count = max(self.tracking_obj_frame_time_count, 1)
        self.tracking_cam_frame_time_count = max(self.tracking_cam_frame_time_count, 1)
        # Compute Average Runtimes
        print(f"Average Object Tracking/Iter Time: {self.tracking_obj_iter_time_sum/self.tracking_obj_iter_time_count} s")
        print(f"Average Cam Tracking/Iter Time: {self.tracking_cam_iter_time_sum/self.tracking_cam_iter_time_count} s")
        print(f"Average Object Tracking/Frame Time: {self.tracking_obj_frame_time_sum/self.tracking_obj_frame_time_count} s")
        print(f"Average Cam Tracking/Frame Time: {self.tracking_cam_frame_time_sum/self.tracking_cam_frame_time_count} s")
        if self.config['use_wandb']:
            self.wandb_run.log({
                        "Final Stats/Average Object Tracking Iter Time (s)": self.tracking_obj_iter_time_sum/self.tracking_obj_iter_time_count,
                        "Final Stats/Average Cam Tracking Iter Time (s)": self.tracking_cam_iter_time_sum/self.tracking_cam_iter_time_count,
                        "Final Stats/Average Object Tracking Frame Time (s)": self.tracking_obj_frame_time_sum/self.tracking_obj_frame_time_count,
                        "Final Stats/Average Cam Tracking Frame Time (s)": self.tracking_cam_frame_time_sum/self.tracking_cam_frame_time_count,
                        "Final Stats/step": 1})

    def log_eval_during(self):
        # Compute Average Metrics
        psnr_list = np.array(self.psnr_list)
        rmse_list = np.array(self.rmse_list)
        l1_list = np.array(self.l1_list)
        ssim_list = np.array(self.ssim_list)
        lpips_list = np.array(self.lpips_list)

        avg_psnr = psnr_list.mean()
        avg_rmse = rmse_list.mean()
        avg_l1 = l1_list.mean()
        avg_ssim = ssim_list.mean()
        avg_lpips = lpips_list.mean()
        print("Average PSNR: {:.2f}".format(avg_psnr))
        print("Average Depth RMSE: {:.2f} cm".format(avg_rmse*100))
        print("Average Depth L1: {:.2f} cm".format(avg_l1*100))
        print("Average MS-SSIM: {:.3f}".format(avg_ssim))
        print("Average LPIPS: {:.3f}".format(avg_lpips))

        if self.config['use_wandb']:
            self.wandb_run.log({"Final Stats/Average PSNR": avg_psnr, 
                            "Final Stats/Average Depth RMSE": avg_rmse,
                            "Final Stats/Average Depth L1": avg_l1,
                            "Final Stats/Average MS-SSIM": avg_ssim, 
                            "Final Stats/Average LPIPS": avg_lpips,
                            "Final Stats/step": 1})

        # Save metric lists as text files
        np.savetxt(os.path.join(self.eval_dir, "psnr.txt"), psnr_list)
        np.savetxt(os.path.join(self.eval_dir, "rmse.txt"), rmse_list)
        np.savetxt(os.path.join(self.eval_dir, "l1.txt"), l1_list)
        np.savetxt(os.path.join(self.eval_dir, "ssim.txt"), ssim_list)
        np.savetxt(os.path.join(self.eval_dir, "lpips.txt"), lpips_list)

    def update(self, psnr, rmse, depth_l1, ssim, lpips):
        self.psnr_list.append(psnr.cpu().numpy())
        self.rmse_list.append(rmse.cpu().numpy())
        self.l1_list.append(depth_l1.cpu().numpy())
        self.ssim_list.append(ssim.cpu().numpy())
        self.lpips_list.append(lpips)