## 核心函数:
# projects/mmdet3d_plugin/datasets/pipelines/loading.py
# projects/mmdet3d_plugin/models/detectors/beverse.py




# projects/mmdet3d_plugin/datasets/pipelines/loading.py
# h63-h258
# 这是test pipeline里执行的第一个方法,目的是从文件中读取到多视角的图片
# 对于tta的作用是loading里边,构建好经过不同增强方式,得到的多组单batch的imgs数据.
# 需要注意的是,tta一般来说只支持一次对一个batch进行处理 也就是 test 的时候,batchsize要设置为1
@PIPELINES.register_module()
class LoadMultiViewImageFromFiles_MTL(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, is_train=False, using_ego=False, temporal_consist=False,
                 data_aug_conf={
                     'resize_lim': (0.193, 0.225),
                     'final_dim': (128, 352),
                     'rot_lim': (-5.4, 5.4),
                     'H': 900, 'W': 1600,
                     'rand_flip': True,
                     'bot_pct_lim': (0.0, 0.22),
                     'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                              'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
                     'Ncams': 5,
                 }, load_seg_gt=False, num_seg_classes=14, select_classes=None):

        self.is_train = is_train
        self.using_ego = using_ego
        self.data_aug_conf = data_aug_conf ## 这个是重点,制定了对于数据增强的基础设置,并且其在config中传进来的值和这里的默认值不一样,应该以传进来的值为准
        self.load_seg_gt = load_seg_gt
        self.num_seg_classes = num_seg_classes
        self.select_classes = range(
            num_seg_classes) if select_classes is None else select_classes

        self.temporal_consist = temporal_consist ## 这个制定了是否要用时序,因为我们的比赛不让用多帧,所以这个时序的融合,我们不能用.
        self.test_time_augmentation = self.data_aug_conf.get('test_aug', False) ## 这个是在定义是否开启TTA 在config文件中 默认是不开启的

    def sample_augmentation(self, specify_resize=None, specify_flip=None): ## 这个函数用来计算这几个图像增强的具体值
        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        fH, fW = self.data_aug_conf['final_dim']
        if self.is_train:
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims

            crop_h = max(0, newH - fH)
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)

            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
        else:   ## 对于tta 我们重点关注test分支 其中specify_resize和specify_flip就是tta定义的多样数据增强
            resize = max(fH / H, fW / W)
            resize = resize + 0.04
            if specify_resize is not None:
                resize = specify_resize

            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = max(0, newH - fH)
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if specify_flip is None else specify_flip
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def choose_cams(self): ## 这里选择相机的做法和BEVDetOCC也不一样
        if self.is_train and self.data_aug_conf['Ncams'] < len(self.data_aug_conf['cams']):
            cams = np.random.choice(self.data_aug_conf['cams'], self.data_aug_conf['Ncams'],
                                    replace=False)
        else:
            cams = self.data_aug_conf['cams']
            cyclist = self.data_aug_conf.get('cyclist', False)
            if cyclist:
                start_id = np.random.choice(np.arange(len(cams)))
                cams = cams[start_id:] + cams[:start_id]
        return cams

    def get_img_inputs(self, results, specify_resize=None, specify_flip=None): ## 这是函数是这个类的核心 其中specify_resize和specify_flip是针对于tta设计的
        img_infos = results['img_info']

        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []

        cams = self.choose_cams()
        if self.temporal_consist: ## 这个是时序的 我们可以忽略这个操作
            cam_augments = {}
            for cam in cams:
                cam_augments[cam] = self.sample_augmentation(
                    specify_resize=specify_resize, specify_flip=specify_flip)

        for frame_id, img_info in enumerate(img_infos): ## 这里是对于多帧进行for循环遍历的 我们只有一帧 也可以忽略
            imgs.append([])
            rots.append([])
            trans.append([])
            intrins.append([])
            post_rots.append([])
            post_trans.append([])

            for cam in cams: ## 接下来对于图片中的每一个相机代表的img分别做处理
                cam_data = img_info[cam]
                filename = cam_data['data_path']
                filename = os.path.join(
                    results['data_root'], filename.split('nuscenes/')[1])

                img = Image.open(filename)

                # img = imageio.imread(filename)
                # img = Image.fromarray(img)

                post_rot = torch.eye(2)
                post_tran = torch.zeros(2)

                intrin = torch.Tensor(cam_data['cam_intrinsic'])
                # extrinsics
                rot = torch.Tensor(cam_data['sensor2lidar_rotation'])
                tran = torch.Tensor(cam_data['sensor2lidar_translation'])

                # 进一步转换到 LiDAR 坐标系
                if self.using_ego:
                    cam2lidar = torch.eye(4)
                    cam2lidar[:3, :3] = torch.Tensor(
                        cam_data['sensor2lidar_rotation'])
                    cam2lidar[:3, 3] = torch.Tensor(
                        cam_data['sensor2lidar_translation'])

                    lidar2ego = torch.eye(4)
                    lidar2ego[:3, :3] = results['lidar2ego_rots']
                    lidar2ego[:3, 3] = results['lidar2ego_trans']

                    cam2ego = lidar2ego @ cam2lidar

                    rot = cam2ego[:3, :3]
                    tran = cam2ego[:3, 3]

                # augmentation (resize, crop, horizontal flip, rotate) ## 这个是重点 在制定五种数据增强的具体数值
                if self.temporal_consist: ## 时序的 忽略掉
                    resize, resize_dims, crop, flip, rotate = cam_augments[cam]
                else: ## 这是我们需要看的
                    # generate augmentation for each time-step, each
                    resize, resize_dims, crop, flip, rotate = self.sample_augmentation(
                        specify_resize=specify_resize, specify_flip=specify_flip)
                ## 在这里通过img_transform函数具体实现了数据增强
                img, post_rot2, post_tran2 = img_transform(img, post_rot, post_tran,
                                                           resize=resize,
                                                           resize_dims=resize_dims,
                                                           crop=crop,
                                                           flip=flip,
                                                           rotate=rotate)

                post_tran = torch.zeros(3)
                post_rot = torch.eye(3)
                post_tran[:2] = post_tran2
                post_rot[:2, :2] = post_rot2

                imgs[frame_id].append(normalize_img(img))
                intrins[frame_id].append(intrin)
                rots[frame_id].append(rot)
                trans[frame_id].append(tran)
                post_rots[frame_id].append(post_rot)
                post_trans[frame_id].append(post_tran)

        # [num_seq, num_cam, ...] ## 接下来把处理好的每个cam的图片stack组合起来 但是这里也涉及了时序 我们不用写的这么复杂
        imgs = torch.stack([torch.stack(x, dim=0) for x in imgs], dim=0)
        rots = torch.stack([torch.stack(x, dim=0) for x in rots], dim=0)
        trans = torch.stack([torch.stack(x, dim=0) for x in trans], dim=0)
        intrins = torch.stack([torch.stack(x, dim=0) for x in intrins], dim=0)
        post_rots = torch.stack([torch.stack(x, dim=0)
                                for x in post_rots], dim=0)
        post_trans = torch.stack([torch.stack(x, dim=0)
                                 for x in post_trans], dim=0)

        return imgs, rots, trans, intrins, post_rots, post_trans ## 这个输出是关键 我们的tta最后得到的imgs 应该是6张图片

    def __call__(self, results): ## 这里是重点,在这里,定义了如何创建多个经过不同数据增强的同一batch的数据
        if (not self.is_train) and self.test_time_augmentation: ## 如果使用tta的话
            results['flip_aug'] = [] ## 这个只是记录一下filp的数值 可以忽略
            results['scale_aug'] = [] ## 这个同上 可忽略
            img_inputs = []
            for flip in self.data_aug_conf.get('tta_flip', [False, ]):
                for scale in self.data_aug_conf.get('tta_scale', [None, ]):
                    results['flip_aug'].append(flip) ## 记录flip数值 可忽略
                    results['scale_aug'].append(scale) ## 记录flip数值 可忽略
                    img_inputs.append(
                        self.get_img_inputs(results, scale, flip)) ## 开始进行重复遍历 将img_inputs的结果叠加进来

            results['img_inputs'] = img_inputs ## 最终的results
        else:
            results['img_inputs'] = self.get_img_inputs(results)

        return results