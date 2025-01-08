import numpy as np
import os


class Logger():
    def __init__(self, config, wandb_run, eval_dir):
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

    @staticmethod
    def numpy_and_save(save_path, input_list):
        input_list = np.array(input_list)
        np.savetxt(save_path, input_list)
        return input_list
    
    def log_final_stats(self, psnr_list, rmse_list, l1_list, ssim_list, lpips_list):
        if len(psnr_list) > 0:
            # Compute Average Metrics
            psnr_list = self.numpy_and_save(os.path.join(self.eval_dir, "psnr.txt"), psnr_list)
            rmse_list = self.numpy_and_save(os.path.join(self.eval_dir, "rmse.txt"), rmse_list)
            l1_list = self.numpy_and_save(os.path.join(self.eval_dir, "l1.txt"), l1_list)
            ssim_list = self.numpy_and_save(os.path.join(self.eval_dir, "ssim.txt"), ssim_list)
            lpips_list = self.numpy_and_save(os.path.join(self.eval_dir, "lpips.txt"), lpips_list)

            print("Average PSNR: {:.2f}".format(psnr_list.mean()))
            print("Average Depth RMSE: {:.2f} cm".format(rmse_list.mean()*100))
            print("Average Depth L1: {:.2f} cm".format(l1_list.mean()*100))
            print("Average MS-SSIM: {:.3f}".format(ssim_list.mean()))
            print("Average LPIPS: {:.3f}".format(lpips_list.mean()))

            if self.wandb_run is not None:
                self.wandb_run.log({"Final Stats/Average PSNR": psnr_list.mean(), 
                            "Final Stats/Average Depth RMSE": rmse_list.mean(),
                            "Final Stats/Average Depth L1": l1_list.mean(),
                            "Final Stats/Average MS-SSIM": ssim_list.mean(), 
                            "Final Stats/Average LPIPS": lpips_list.mean(),
                            "Final Stats/step": 1})
    
    def report_loss(
            self,
            losses,
            wandb_run,
            wandb_step,
            cam_tracking=False,
            obj_tracking=False,
            delta_optim=False,
            refine=False):

        # Update loss dict
        if cam_tracking:
            tracking_loss_dict = {}
            for k, v in losses.items():
                tracking_loss_dict[f"Per Iteration Cam Tracking/{k}"] = v.item()
            tracking_loss_dict['Per Iteration Cam Tracking/step'] = wandb_step
            wandb_run.log(tracking_loss_dict)
        elif obj_tracking:
            tracking_loss_dict = {}
            for k, v in losses.items():
                tracking_loss_dict[f"Per Iteration Object Tracking/{k}"] = v.item()
            tracking_loss_dict['Per Iteration Object Tracking/step'] = wandb_step
            wandb_run.log(tracking_loss_dict)
        elif refine:
            tracking_loss_dict = {}
            for k, v in losses.items():
                tracking_loss_dict[f"Per Iteration Refine/{k}"] = v.item()
            tracking_loss_dict['Per Iteration Refine/step'] = wandb_step
            wandb_run.log(tracking_loss_dict)
        elif delta_optim:
            delta_loss_dict = {}
            for k, v in losses.items():
                delta_loss_dict[f"Per Iteration Delta Optim/{k}"] = v.item()
            delta_loss_dict['Per Iteration Delta Optim/step'] = wandb_step
            wandb_run.log(delta_loss_dict)
        
        # Increment wandb step
        wandb_step += 1
        return wandb_step