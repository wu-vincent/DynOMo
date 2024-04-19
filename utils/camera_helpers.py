import torch
try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera
except:
    pass


def get_camera_params(config):
    h, w = config["data"]["desired_image_height"], config["data"]["desired_image_width"]


def get_projection_matrix(w, h, k, w2c, near=0.01, far=100, only_proj=True):
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    w2c = torch.tensor(w2c).cuda().float()
    cam_center = torch.inverse(w2c)[:3, 3]
    w2c = w2c.unsqueeze(0).transpose(1, 2)
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
    full_proj = w2c.bmm(opengl_proj)
    if not only_proj:
        return full_proj, fx, fy, cam_center, w2c
    else:
        return full_proj

def setup_camera(w, h, k, w2c, near=0.01, far=100):
    full_proj, fx, fy, cam_center, w2c = get_projection_matrix(
        w, h, k, w2c, near, far, only_proj=False)
    cam = Camera(
        image_height=h,
        image_width=w,
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=w2c,
        projmatrix=full_proj,
        sh_degree=0,
        campos=cam_center,
        prefiltered=False
    )
    return cam
