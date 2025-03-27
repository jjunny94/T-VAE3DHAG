import models.motion_trans as vae_models
from trainer.vae_trainer import *
from utils.plot_script import *
import utils.paramUtil as paramUtil
from utils.utils_ import *
from options.evaluate_vae_options import *
from dataProcessing import dataset
from torch.utils.data import DataLoader

if __name__ == "__main__":
    parser = TestOptions()
    opt = parser.parse()
    joints_num = 0
    input_size = 72
    data = None
    label_dec = None
    dim_category = 31
    enumerator = None
    device = torch.device("cuda:" + str(opt.gpu_id) if opt.gpu_id else "cpu")

    opt.save_root = os.path.join(opt.checkpoints_dir, opt.dataset_type, opt.name)
    opt.model_path = os.path.join(opt.save_root, 'model')
    opt.joints_path = os.path.join(opt.save_root, 'joints')

    model_file_path = os.path.join(opt.model_path, opt.which_epoch + '.tar')
    result_path = os.path.join(opt.result_path, opt.dataset_type, opt.name + opt.name_ext)

    if opt.dataset_type == "humanact12":
        dataset_path = "./dataset/humanact12"
        input_size = 72
        joints_num = 24
        label_dec = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        raw_offsets = paramUtil.humanact12_raw_offsets
        kinematic_chain = paramUtil.humanact12_kinematic_chain
        enumerator = paramUtil.humanact12_coarse_action_enumerator
    else:
        raise NotImplementedError('This dataset is unregonized!!!')

    opt.dim_category = len(label_dec)

    opt.pose_dim = input_size

    if opt.time_counter:
        opt.input_size = input_size + opt.dim_category + 1
    else:
        opt.input_size = input_size + opt.dim_category

    opt.output_size = input_size

    model = torch.load(model_file_path, map_location='cuda:0')
    prior_net = vae_models.Encoder(opt.input_size, opt.dim_z, opt.hidden_size, 8,
                                       opt.prior_hidden_layers, opt.num_samples, device)

    decoder = vae_models.Decoder(opt.input_size + opt.dim_z, opt.output_size, opt.hidden_size, 8,
                                 opt.decoder_hidden_layers,
                                 opt.num_samples, device)

    prior_net.load_state_dict(model['prior_net'])
    decoder.load_state_dict(model['decoder'])
    prior_net.to(device)
    decoder.to(device)
    if opt.use_lie:
        if opt.dataset_type == 'humanact12':
            data = dataset.MotionFolderDatasetHumanAct12(dataset_path, opt, lie_enforce=opt.lie_enforce)
        motion_dataset = dataset.MotionDataset(data, opt)
        motion_loader = DataLoader(motion_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=2,
                                   shuffle=True)
        trainer = TrainerLie(motion_loader, opt, device, raw_offsets, kinematic_chain)
    else:
        trainer = Trainer(None, opt, device)

    if opt.do_random:
        fake_motion, classes = trainer.evaluate(prior_net, decoder, opt.num_samples)
        fake_motion = fake_motion.cpu().numpy()
    else:
        categories = np.arange(opt.dim_category).repeat(opt.replic_times, axis=0)
        num_samples = categories.shape[0]
        category_oh, classes = trainer.get_cate_one_hot(categories)
        fake_motion, _ = trainer.evaluate(prior_net, decoder, num_samples, category_oh)
        fake_motion = fake_motion.cpu().numpy()

    print(fake_motion.shape)
    for i in range(fake_motion.shape[0]):
        class_type = enumerator[label_dec[classes[i]]]
        motion_orig = fake_motion[i]
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        keypoint_path = os.path.join(result_path, 'keypoint')
        if not os.path.exists(keypoint_path):
            os.makedirs(keypoint_path)
        file_name = os.path.join(result_path, class_type + str(i) + ".gif")
        offset = np.matlib.repmat(np.array([motion_orig[0, 0], motion_orig[0, 1], motion_orig[0, 2]]),
                                     motion_orig.shape[0], joints_num)

        motion_mat = motion_orig - offset

        motion_mat = motion_mat.reshape(-1, joints_num, 3)
        #print(motion_mat)
        np.save(os.path.join(keypoint_path, class_type + str(i) + '_3d.npy'), motion_mat)

        if opt.dataset_type == "humanact12":
            plot_3d_motion_v2(motion_mat, kinematic_chain, save_path=file_name, interval=80)

