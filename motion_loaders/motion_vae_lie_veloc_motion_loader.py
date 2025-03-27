import torch
import models.motion_trans as vae_models
# import models.networks as networks
from trainer.vae_trainer import *
from torch.utils.data import Dataset
import utils.paramUtil as paramUtil
from dataProcessing import dataset
from torch.utils.data import DataLoader

class MotionVAEVelocGeneratedDataset(Dataset):
    def __init__(self, opt, num_motions, batch_size, device, ground_motion_loader, label=None):
        if opt.dataset_type == 'ntu_rgbd_vibe':
            raw_offsets = paramUtil.vibe_raw_offsets
            kinematic_chain = paramUtil.vibe_kinematic_chain
        elif (opt.dataset_type == 'humanact12') or (opt.dataset_type == 'humanact12'):
            raw_offsets = paramUtil.humanact12_raw_offsets
            kinematic_chain = paramUtil.humanact12_kinematic_chain
        elif opt.dataset_type == 'mocap':
            raw_offsets = paramUtil.mocap_raw_offsets
            kinematic_chain = paramUtil.mocap_kinematic_chain
        else:
            raise NotImplementedError("Data type not Found")
        print(opt.model_file_path)
        model = torch.load(opt.model_file_path)

        self.prior_net = vae_models.GaussianGRU(opt.input_size, opt.dim_z, opt.hidden_size, 8,
                                           opt.prior_hidden_layers, opt.batch_size, device)
        self.posterior_net = vae_models.GaussianGRU(opt.input_size, opt.dim_z, opt.hidden_size, 8,
                                               opt.posterior_hidden_layers, opt.batch_size, device)
        # if opt.use_vel_S:
        #     self.veloc_net = networks.VelocityNetwork_Sim(opt.veloc_input_size, 3, opt.hidden_size)
        # else:
        #     self.veloc_net = networks.VelocityNetwork(opt.veloc_input_size, 3, opt.hidden_size, opt.veloc_hidden_layers,
                                                    #   opt.batch_size, device)

        # self.decoder = vae_models.DecoderGRU(opt.input_size + opt.dim_z, opt.output_size, opt.hidden_size, 8,
        #                                      opt.decoder_hidden_layers, opt.batch_size, device, use_hdl=opt.use_hdl,
        #                                           do_all_parent=opt.do_all_parent, kinematic_chains=kinematic_chain)
        self.decoder = vae_models.DecoderGRU(opt.input_size + opt.dim_z, opt.output_size, opt.hidden_size, 8,
                                        opt.decoder_hidden_layers,
                                        opt.num_samples, device)

        self.prior_net.load_state_dict(model['prior_net'])
        # self.veloc_net.load_state_dict(model['veloc_net'])
        self.decoder.load_state_dict(model['decoder'])

        self.prior_net.to(device)
        # self.veloc_net.to(device)
        self.decoder.to(device)

        self.num_motions = num_motions
        # if opt.do_relative:
        #     self.trainer = TrainerLieV3(None, opt, device, raw_offsets, kinematic_chain)
        # else:
        #     self.trainer = TrainerLieV2(None, opt, device, raw_offsets, kinematic_chain)

        self.pool_size = num_motions
        self.resize_counter = 0
        self.opt = opt
        self.batch_size = batch_size
        self.ground_motion_loader = ground_motion_loader
        self.label = label
        self.initiatize(self.opt, self.pool_size, self.batch_size, self.ground_motion_loader)
        

    def initiatize(self, opt, num_motions, batch_size, ground_motion_loader):
        self.resize_counter += 1
        #print_opt = vars(opt)
        #for k, v in sorted(print_opt.items()):
        #    print("%s: %s" % (k, v))
        dataset_path = "./dataset/humanact12"
        input_size = 72
        joints_num = 24
        label_dec = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        data = dataset.MotionFolderDatasetHumanAct12(dataset_path, opt, lie_enforce=opt.lie_enforce)
        raw_offsets = paramUtil.humanact12_raw_offsets
        kinematic_chain = paramUtil.humanact12_kinematic_chain
        enumerator = paramUtil.humanact12_coarse_action_enumerator
        motion_dataset = dataset.MotionDataset(data, opt)
        motion_loader = DataLoader(motion_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=2,
                                   shuffle=True)
        device = torch.device("cuda:" + str(opt.gpu_id) if opt.gpu_id else "cpu")
        trainer = TrainerLie(motion_loader, opt, device, raw_offsets, kinematic_chain)
        if opt.do_random:
            motions_output = torch.zeros(num_motions, opt.motion_length, opt.input_size_raw).numpy()
            labels_output = torch.zeros(num_motions).numpy()
            real_iter = iter(ground_motion_loader)
            while num_motions:
                num_motions_batch = min(batch_size, num_motions)
                real_joints_list = []
                while len(real_joints_list) < num_motions_batch:
                    real_joints_list.append(next(real_iter)[0])
                real_joints = torch.cat(real_joints_list, dim=0)

                opt.num_samples = num_motions_batch
                cate_one_hot = None
                if self.label is not None:
                    categories = np.ones(opt.num_samples, dtype=np.int)
                    categories.fill(self.label)
                    cate_one_hot, _ = trainer.get_cate_one_hot(categories)
                motions_output_batch, labels_output_batch = \
                    trainer.evaluate(self.prior_net, self.decoder, opt.num_samples,
                                          cate_one_hot=cate_one_hot, real_joints=real_joints)
                if self.label is not None:
                    labels_output_batch = self.label
                motions_output[(num_motions-num_motions_batch):num_motions, :, :] = motions_output_batch
                labels_output[(num_motions-num_motions_batch):num_motions] = labels_output_batch

                num_motions -= num_motions_batch

        else:
            raise NotImplementedError('LOL, not today!')

        self.motions_output = motions_output
        self.labels_output = labels_output

    def __len__(self):
        return self.num_motions

    def __getitem__(self, item):
        if (item / self.pool_size) >= self.resize_counter:
            self.initiatize(self.opt, self.pool_size, self.batch_size, self.ground_motion_loader)
        item = item % self.pool_size
        return self.motions_output[item, :, :], self.labels_output[item]
