import torch.optim as optim
import random
from collections import OrderedDict
from utils.utils_ import *
from lie.pose_lie import *
from lie.lie_util import *
from utils.paramUtil import *

# Trainer for model without Lie
class Trainer(object):
    def __init__(self, motion_sampler, opt, device):
        self.opt = opt
        self.device = device
        self.motion_sampler = motion_sampler
        self.motion_enumerator = None
        self.opt_generator = None
        if self.opt.isTrain:
            self.align_criterion = nn.MSELoss()
            self.recon_criterion = nn.MSELoss()

    def ones_like(self, t, val=1):
        return torch.Tensor(t.size()).fill_(val).requires_grad_(False).to(self.device)

    def zeros_like(self, t, val=0):
        return torch.Tensor(t.size()).fill_(val).requires_grad_(False).to(self.device)

    def tensor_fill(self, tensor_size, val=0):
        return torch.zeros(tensor_size).fill_(val).requires_grad_(False).to(self.device)

    def sample_real_motion_batch(self):
        if self.motion_enumerator is None:
            self.motion_enumerator = enumerate(self.motion_sampler)

        batch_idx, batch = next(self.motion_enumerator)
        if batch_idx == len(self.motion_sampler) - 1:
            self.motion_enumerator = enumerate(self.motion_sampler)
        self.real_motion_batch = batch
        return batch

    def kl_criterion(self, mu1, logvar1, mu2, logvar2):
        # KL( N(mu1, sigma2_1) || N(mu_2, sigma2_2))
        # loss = log(sigma2/sigma1) / 2 + (sigma1 + (mu1 - mu2)^2)/(2*sigma2) - 1/2
        sigma1 = logvar1.mul(0.5).exp()
        sigma2 = logvar2.mul(0.5).exp()
        kld = torch.log(sigma2/sigma1) + (torch.exp(logvar1) + (mu1-mu2)**2)/(2*torch.exp(logvar2)) - 1/2
        return kld.sum() / self.opt.batch_size

    def sample_z_cate(self, batch_size):
        if self.opt.dim_category <= 0:
            return None, np.zeros(batch_size)
        # dim (num_samples, )
        classes_to_generate = np.random.randint(self.opt.dim_category, size=batch_size)
        # dim (num_samples, dim_category)
        one_hot = np.zeros((classes_to_generate.shape[0], self.opt.dim_category), dtype=np.float32)
        one_hot[np.arange(classes_to_generate.shape[0]), classes_to_generate] = 1

        # dim (num_samples, dim_category)
        one_hot_motion = torch.from_numpy(one_hot).to(self.device).requires_grad_(False)

        return one_hot_motion, classes_to_generate

    def get_cate_one_hot(self, categories):
        classes_to_generate = np.array(categories).reshape((-1,))
        # dim (num_samples, dim_category)
        one_hot = np.zeros((categories.shape[0], self.opt.dim_category), dtype=np.float32)
        one_hot[np.arange(categories.shape[0]), classes_to_generate] = 1

        # dim (num_samples, dim_category)
        one_hot_motion = torch.from_numpy(one_hot).to(self.device).requires_grad_(False)

        return one_hot_motion, classes_to_generate


    def train(self, prior_net, posterior_net, decoder, opt_prior_net, opt_posterior_net, opt_decoder, sample_true):
        opt_prior_net.zero_grad()
        opt_posterior_net.zero_grad()
        opt_decoder.zero_grad()

        prior_net.init_hidden()
        posterior_net.init_hidden()
        decoder.init_hidden()
        # data(batch_size, motion_len, joints_num * 3)
        data, cate_data = sample_true()
        self.real_data = data
        # dim(batch_size, category_dim)
        cate_one_hot, classes_to_generate = self.get_cate_one_hot(cate_data)
        data = torch.clone(data).float().detach_().to(self.device)
        motion_length = data.shape[1]
        # print(data.shape[0], data.shape[2])
        # dim(batch_size, pose_dim), initial prior is a zero vector
        prior_vec = self.tensor_fill((data.shape[0], data.shape[2]), 0)
        # print(prior_vec.shape)
        log_dict = OrderedDict({'g_loss': 0})

        teacher_force = True if random.random() < self.opt.tf_ratio else False
        mse = 0
        kld = 0

        opt_step_cnt = 0

        for i in range(0, motion_length):
            condition_vec = cate_one_hot
            if self.opt.time_counter:
                time_counter = i / (motion_length - 1)
                time_counter_vec = self.tensor_fill((data.shape[0], 1), time_counter)
                condition_vec = torch.cat((cate_one_hot, time_counter_vec), dim=1)
            # print(prior_vec.shape, condition_vec.shape)
            # print("qwerqwtwqetgadsr")
            # print(prior_vec.shape, condition_vec.shape, data.shape[0], data.shape[2])
            h = torch.cat((prior_vec, condition_vec), dim=1)
            h_target = torch.cat((data[:, i], condition_vec), dim=1)

            z_t, mu, logvar, h_in_p, f_o_o = posterior_net(h_target)
            _, mu_p, logvar_p, _, f_o = prior_net(h)
            # print(h.shape, z_t.shape)
            h_mid = torch.cat((h, z_t), dim=1)
            x_pred, h_in = decoder(h_mid, f_o)
            # print(mu.shape, logvar.shape, mu_p.shape, logvar_p.shape)
            # whether to skip the optimization of current time step(not used in our paper)
            is_skip = True if random.random() < self.opt.skip_prob else False
            if not is_skip:
                opt_step_cnt += 1
                mse += self.recon_criterion(x_pred, data[:, i])
                # print(data[:, i].shape)
                kld += self.kl_criterion(mu, logvar, mu_p, logvar_p)
            if teacher_force:
                prior_vec = x_pred
            else:
                prior_vec = data[:, i]

        log_dict['g_recon_loss'] = mse.item() / opt_step_cnt
        log_dict['g_kld_loss'] = kld.item() / opt_step_cnt
        losses = mse + kld * self.opt.lambda_kld

        avg_loss = losses.item() / opt_step_cnt
        losses.backward()

        opt_prior_net.step()
        opt_posterior_net.step()
        opt_decoder.step()
        log_dict['g_loss'] = avg_loss

        return log_dict

    def evaluate(self, prior_net, decoder, num_samples, cate_one_hot=None):
        prior_net.eval()
        decoder.eval()
        with torch.no_grad():
            if cate_one_hot is None:
                cate_one_hot, classes_to_generate = self.sample_z_cate(num_samples)
            else:
                classes_to_generate = None
            prior_vec = self.tensor_fill((num_samples, self.opt.pose_dim), 0)
            # prior_net.init_hidden(num_samples)
            # decoder.init_hidden(num_samples)

            generate_batch = []
            for i in range(0, self.opt.motion_length):
                condition_vec = cate_one_hot
                if self.opt.time_counter:
                    time_counter = i / (self.opt.motion_length - 1)
                    time_counter_vec = self.tensor_fill((num_samples, 1), time_counter)
                    condition_vec = torch.cat((cate_one_hot, time_counter_vec), dim=1)
                # print(prior_vec.shape, condition_vec.shape)
                h = torch.cat((prior_vec, condition_vec), dim=1)
                z_t_p, mu_p, logvar_p, h_in_p, f_o = prior_net(h)

                h_mid = torch.cat((h, z_t_p), dim=1)
                x_pred, _ = decoder(h_mid, f_o)
                prior_vec = x_pred
                generate_batch.append(x_pred.unsqueeze(1))


            generate_batch = torch.cat(generate_batch, dim=1)

        return generate_batch.cpu(), classes_to_generate

    def trainIters(self, prior_net, posterior_net, decoder):
        self.opt_decoder = optim.Adam(decoder.parameters(), lr=0.0002, betas=(0.9, 0.999), weight_decay=0.00001)
        self.opt_prior_net = optim.Adam(prior_net.parameters(), lr=0.0002, betas=(0.9, 0.999), weight_decay=0.00001)
        self.opt_posterior_net = optim.Adam(posterior_net.parameters(), lr=0.0002, betas=(0.9, 0.999), weight_decay=0.00001)

        prior_net.to(self.device)
        posterior_net.to(self.device)
        decoder.to(self.device)

        def save_model(file_name):
            state = {
                "prior_net": prior_net.state_dict(),
                "posterior_net": posterior_net.state_dict(),
                "decoder": decoder.state_dict(),
                "opt_prior_net": self.opt_prior_net.state_dict(),
                "opt_posterior_net": self.opt_posterior_net.state_dict(),
                "opt_decoder": self.opt_decoder.state_dict(),
                "iterations": iter_num
            }

            torch.save(state, os.path.join(self.opt.model_path, file_name + ".tar"))

        def load_model(file_name):
            model = torch.load(os.path.join(self.opt.model_path, file_name + '.tar'))
            prior_net.load_state_dict(model['prior_net'])
            posterior_net.load_state_dict(model['posterior_net'])
            decoder.load_state_dict(model['decoder'])
            self.opt_prior_net.load_state_dict(model['opt_prior_net'])
            self.opt_posterior_net.load_state_dict(model['opt_posterior_net'])
            self.opt_decoder.load_state_dict(model['opt_decoder'])


        if self.opt.is_continue and self.opt.isTrain:
            load_model('latest')

        iter_num = 0
        logs = OrderedDict()
        start_time = time.time()

        e_num_samples = 20
        cate_one_hot, classes = self.sample_z_cate(e_num_samples)
        np.save(os.path.join(self.opt.joints_path, "motion_class.npy"), classes)

        while True:
            prior_net.train()
            posterior_net.train()
            decoder.train()

            gen_log_dict = self.train(prior_net, posterior_net, decoder, self.opt_prior_net, self.opt_posterior_net,
                                      self.opt_decoder,
                                      self.sample_real_motion_batch )

            for k, v in gen_log_dict.items():
                if k not in logs:
                    logs[k] = [v]
                else:
                    logs[k].append(v)

            iter_num += 1

            if iter_num % self.opt.print_every == 0:
                mean_loss = OrderedDict()
                for k, v in logs.items():
                    mean_loss[k] = sum(logs[k][-1 * self.opt.print_every:]) / self.opt.print_every
                print_current_loss(start_time, iter_num, self.opt.iters, mean_loss)

            if iter_num % self.opt.eval_every == 0:
                fake_motion, _ = self.evaluate(prior_net, decoder, e_num_samples, cate_one_hot)
                np.save(os.path.join(self.opt.joints_path, "motion_joints" + str(iter_num) + ".npy"), fake_motion)

            if iter_num % self.opt.save_every == 0:
                save_model(str(iter_num))

            if iter_num % self.opt.save_latest == 0:
                save_model('latest')

            if iter_num >= self.opt.iters:
                break
        return logs

# trainning with lie algebra paramters
class TrainerLie(Trainer):
    def __init__(self, motion_sampler, opt, device, raw_offsets, kinematic_chain):
        super(TrainerLie, self).__init__(motion_sampler,
                                         opt,
                                         device)
        self.raw_offsets = torch.from_numpy(raw_offsets).to(device).detach()
        self.kinematic_chain = kinematic_chain
        self.Tensor = torch.Tensor if self.opt.gpu_id is None else torch.cuda.FloatTensor
        self.lie_skeleton = LieSkeleton(self.raw_offsets, kinematic_chain, self.Tensor)
        if self.opt.isTrain:

            if self.opt.lie_enforce:
                # not used in our paper
                self.mse_lie = nn.MSELoss()
                self.mse_trajec = nn.MSELoss()
                if self.opt.use_geo_loss:
                    self.recon_criterion = self.geo_loss
                else:
                    self.recon_criterion = self.weight_mse_loss
            else:
                self.mse = nn.MSELoss()
                self.recon_criterion = self.mse_lie

    def geo_loss(self, lie_param1, lie_param2):
        # lie_param (batch_size, joints_num*3)
        joints_num = int(lie_param1.shape[-1] / 3)
        # lie_al1 (batch_size, joints_num - 1, 3)
        lie_al1 = lie_param1[..., 3:].view(-1, joints_num - 1, 3)
        lie_al2 = lie_param2[..., 3:].view(-1, joints_num - 1, 3)
        # root_trans (batch_size, 3)
        root_trans1 = lie_param1[..., :3]
        root_trans2 = lie_param2[..., :3]
        # rot mat (batch_size, joints_num-1, 3, 3)
        rot_mat1 = lie_exp_map(lie_al1)
        rot_mat2 = lie_exp_map(lie_al2)
        rm1_rm2_T = torch.matmul(rot_mat1, rot_mat2.transpose(2, 3))
        rm1_T_rm2 = torch.matmul(rot_mat1.transpose(2, 3), rot_mat2)
        log_map = (rm1_rm2_T - rm1_T_rm2) / 2
        # A (batch_size, joints_num, 3)
        A = torch.cat((log_map[..., 2, 1, None],
                       log_map[..., 0, 2, None],
                       log_map[..., 1, 0, None]),
                      dim=-1)
        geo_dis = torch.mul(A, A).sum(dim=-1)
        geo_dis = (geo_dis**2).sum()
        # root trans loss
        rt_dis = self.mse_trajec(root_trans1, root_trans2)
        return geo_dis + self.opt.lambda_trajec * rt_dis

    def weight_mse_loss(self, lie_param1, lie_param2):
        # lie_param (batch_size, joints_num*3)
        # lie_al1 (batch_size, (joints_num - 1)*3)
        lie_al1 = lie_param1[..., 3:]
        lie_al2 = lie_param2[..., 3:]
        # root_trans (batch_size, 3)
        root_trans1 = lie_param1[..., :3]
        root_trans2 = lie_param2[..., :3]

        return self.mse_lie(lie_al1, lie_al2) +\
               self.opt.lambda_trajec * self.mse_trajec(root_trans1, root_trans2)

    def mse_lie(self, lie_param, target_joints):
        # use the target joints to calculate bone length
        real_joints = target_joints
        generated_joints = self.pose_lie_2_joints(lie_param, real_joints)
        return self.mse(generated_joints, real_joints)

    # Transform Lie parameters to 3d coordinates
    def pose_lie_2_joints(self, lie_batch, pose_batch):
        if self.opt.no_trajectory:
            lie_params = lie_batch
            root_translation = self.zeros_like(lie_batch[..., :3], 0)
        else:
            lie_params = lie_batch[..., 3:]
            root_translation = lie_batch[..., :3]
        zero_padding = self.zeros_like(root_translation, 0)
        lie_params = torch.cat((zero_padding, lie_params), dim=-1)
        num_samples = pose_batch.shape[0]
        pose_batch = pose_batch.view(num_samples, -1, 3)
        pose_joints = self.lie_to_joints(lie_params, pose_batch, root_translation)
        return pose_joints

    def lie_to_joints(self, lie_params, joints, root_translation):
        lie_params = lie_params.view(lie_params.shape[0], -1, 3)
        joints = self.lie_skeleton.forward_kinematics(lie_params, joints, root_translation)
        return joints.view(joints.shape[0], -1)

    def lie_to_joints_v2(self, lie_params, joints, root_translation, scale_inds):
        lie_params = lie_params.view(lie_params.shape[0], -1, 3)
        joints = self.lie_skeleton.forward_kinematics(lie_params, joints, root_translation, scale_inds=scale_inds)
        return joints.view(joints.shape[0], -1)

    def evaluate(self, prior_net, decoder, num_samples, cate_one_hot=None, real_joints=None):
        generated_batch, classes_to_generate = super(TrainerLie, self).evaluate(
            prior_net, decoder, num_samples, cate_one_hot)
        if not self.opt.isTrain:
            generated_batch_lie = generated_batch.to(self.device)
            #real_joints (batch_size, motion_length, joint_num*3)
            if real_joints is None:
                real_joints, cate_data = self.sample_real_motion_batch()
            if real_joints.shape[0] < num_samples:
                repeat_ratio = int(num_samples / real_joints.shape[0])
                real_joints = real_joints.repeat((repeat_ratio, 1, 1))
                pad_num = num_samples - real_joints.shape[0]
                if pad_num != 0:
                    real_joints = torch.cat((real_joints, real_joints[: pad_num]), dim=0)
            else:
                real_joints = real_joints[:num_samples]
            # (batch_size, motion_length, joints_num, 3)
            real_joints = real_joints[:, 0, :].view(num_samples, -1, 3)
            real_joints = self.Tensor(real_joints.size()).copy_(real_joints)
            generated_batch = []
            for i in range(self.opt.motion_length):
                # (batch_size, joints_num, 3)
                joints_batch = self.lie_to_joints(generated_batch_lie[:, i, :], real_joints, generated_batch_lie[:, i, :3])
                joints_batch = joints_batch.unsqueeze(1)
                generated_batch.append(joints_batch)
            generated_batch = torch.cat(generated_batch, dim=1)

        return generated_batch.cpu(), classes_to_generate


    # Evaluation with variable bone lengths
    def evaluate3(self, prior_net, decoder, num_samples, cate_one_hot=None, real_joints=None):
        generated_batch, classes_to_generate = super(TrainerLie, self).evaluate(
            prior_net, decoder, num_samples, cate_one_hot)
        kinematic_chains = humanact12_kinematic_chain
        if not self.opt.isTrain:
            generated_batch_lie = generated_batch.to(self.device)
            #real_joints (batch_size, motion_length, joint_num*3)
            if real_joints is None:
                real_joints, cate_data = self.sample_real_motion_batch()
            batch_size = real_joints.shape[0]
            li = [real_joints[i].repeat(num_samples, 1, 1) for i in range(real_joints.shape[0])]
            real_joints = torch.cat(li, dim=0)
            real_joints1 = real_joints.clone()
            real_joints2 = real_joints.clone()
            real_joints3 = real_joints.clone()
            leg_indx = kinematic_chains[0] + kinematic_chains[1]
            arm_indx = kinematic_chains[3] + kinematic_chains[4]
            all_indx = [i for i in range(24)]
            scale_list = [None, leg_indx, arm_indx, all_indx]
            # scale_list = [None]
            generated_batch_lie = generated_batch_lie.repeat(batch_size, 1, 1)
            classes_to_generate = np.tile(classes_to_generate, batch_size)
            num_samples = generated_batch_lie.shape[0]
            # (batch_size, motion_length, joints_num, 3)
            real_joints = real_joints[:, 0, :].view(num_samples, -1, 3)
            real_joints = real_joints.to(self.device)
            generated_batch_list = []
            for scale in scale_list:
                generated_batch = []
                for i in range(self.opt.motion_length):
                    # (batch_size, joints_num, 3)
                    joints_batch = self.lie_to_joints_v2(generated_batch_lie[:, i, :], real_joints, generated_batch_lie[:, i, :3], scale)
                    joints_batch = joints_batch.unsqueeze(1)
                    generated_batch.append(joints_batch)
                generated_batch = torch.cat(generated_batch, dim=1)
                generated_batch_list.append(generated_batch)

        return torch.cat(generated_batch_list, dim=0).cpu(), np.tile(classes_to_generate, len(scale_list))
    

class TrainerLieV2(Trainer):
    def __init__(self, motion_sampler, opt, device, raw_offsets, kinematic_chain):
        super(TrainerLieV2, self).__init__(motion_sampler,
                                           opt,
                                           device)

        self.raw_offsets = torch.from_numpy(raw_offsets).to(device).detach()
        self.kinematic_chain = kinematic_chain
        self.Tensor = torch.Tensor if self.opt.gpu_id is None else torch.cuda.FloatTensor
        self.lie_skeleton = LieSkeleton(self.raw_offsets, kinematic_chain, self.Tensor)
        if self.opt.isTrain:
            # self.recon_criterion = nn.MSELoss()
            self.l2_trajec = nn.MSELoss()

    def recon_criterion(self, pred_vec, ground_vec):
        return nn.MSELoss(pred_vec, ground_vec)

    def train(self, prior_net, posterior_net, decoder, veloc_net,
              opt_prior_net, opt_posterior_net, opt_decoder, opt_veloc_net, sample_true):
        opt_prior_net.zero_grad()
        opt_posterior_net.zero_grad()
        opt_decoder.zero_grad()
        opt_veloc_net.zero_grad()

        prior_net.init_hidden()
        posterior_net.init_hidden()
        decoder.init_hidden()
        veloc_net.init_hidden()

        # data(batch_size, motion_len, joints_num * 3)
        data, cate_data = sample_true()
        self.real_data = data
        # dim(batch_size, category_dim)
        cate_one_hot, classes_to_generate = self.get_cate_one_hot(cate_data)
        data = self.Tensor(data.size()).copy_(data).detach_()
        motion_length = data.shape[1]
        # dim(batch_size, pose_dim), initial prior is a zero vector
        prior_vec = self.tensor_fill((data.shape[0], data.shape[2]), 0)

        log_dict = OrderedDict({'g_loss': 0})

        teacher_force = True if random.random() < self.opt.tf_ratio else False
        mse = 0
        kld = 0
        trajc_align = 0
        opt_step_cnt = 0

        for i in range(0, motion_length):
            condition_vec = cate_one_hot
            if self.opt.time_counter:
                time_counter = i / (motion_length - 1)
                time_counter_vec = self.tensor_fill((data.shape[0], 1), time_counter)
                condition_vec = torch.cat((cate_one_hot, time_counter_vec), dim=1)
            # print(prior_vec.shape, condition_vec.shape)
            h = torch.cat((prior_vec, condition_vec), dim=1)
            h_target = torch.cat((data[:, i], condition_vec), dim=1)

            z_t, mu, logvar, h_in_p = posterior_net(h_target)
            _, mu_p, logvar_p, _ = prior_net(h)

            h_mid = torch.cat((h, z_t), dim=1)
            lie_out, vel_mid, h_in = decoder(h_mid)
            joints_o_traj = self.pose_lie_2_joints(lie_out, data[:, i], i == 0)
            if i == 0:
                pred_joints = joints_o_traj
            else:
                prior_traj = prior_vec[:, :3].detach()
                prior_o_traj = prior_vec.detach() - prior_traj.repeat(1, int(prior_vec.shape[1]/3))
                if self.opt.use_vel_H:
                    vel_in = (prior_o_traj, joints_o_traj, vel_mid)
                else:
                    vel_in = torch.cat((prior_o_traj, joints_o_traj, vel_mid), dim=1)
                vel_out = veloc_net(vel_in)
                trajec = prior_traj + vel_out
                pred_joints = joints_o_traj + trajec.repeat(1, int(joints_o_traj.shape[1]/3))
            is_skip = True if random.random() < self.opt.skip_prob else False
            if not is_skip:
                opt_step_cnt += 1
                kld += self.kl_criterion(mu, logvar, mu_p, logvar_p)
                if self.opt.optim_seperate:
                    ground_o_traj = data[:, i] - data[:, i, :3].repeat(1, int(data[:, i].shape[1]/3))
                    mse += self.recon_criterion(joints_o_traj, ground_o_traj)
                    if i != 0:
                        ground_vel = data[:, i, :3] - prior_traj
                        trajc_align += self.l2_trajec(vel_out, ground_vel)
                else:
                    mse += self.recon_criterion(pred_joints, data[:, i])
                    if self.opt.do_trajec_align and i != 0:
                        ground_vel = data[:, i, :3] - prior_traj
                        trajc_align += self.l2_trajec(vel_out, ground_vel)
            # generate_batch.append(x_pred.unsqueeze(1))
            if teacher_force:
                prior_vec = pred_joints
            else:
                prior_vec = data[:, i]

        log_dict['g_recon_loss'] = mse.item() / opt_step_cnt
        log_dict['g_kld_loss'] = kld.item() / opt_step_cnt
        losses = mse + kld * self.opt.lambda_kld

        if self.opt.do_trajec_align or self.opt.optim_seperate:
            losses += trajc_align * self.opt.lambda_trajec
            log_dict['g_trajec_align_loss'] = trajc_align.item() / opt_step_cnt

        avg_loss = losses.item() / opt_step_cnt

        losses.backward()

        opt_prior_net.step()
        opt_posterior_net.step()
        opt_decoder.step()
        opt_veloc_net.step()
        log_dict['g_loss'] = avg_loss

        return log_dict

    def pose_lie_2_joints(self, lie_batch, pose_batch, init_pose=False):
        root_translation = torch.zeros((lie_batch.shape[0], 3), requires_grad=False).to(self.device)
        num_samples = pose_batch.shape[0]
        pose_batch = pose_batch.view(num_samples, -1, 3)
        pose_joints = self.lie_to_joints(lie_batch, pose_batch, root_translation, init_pose)
        return pose_joints

    def lie_to_joints(self, lie_params, joints, root_translation, init_pose=False):
        lie_params = lie_params.view(lie_params.shape[0], -1, 3)
        joints = self.lie_skeleton.forward_kinematics(lie_params, joints, root_translation, do_root_R=not init_pose)
        return joints.view(joints.shape[0], -1)

    def evaluate(self, prior_net, decoder, veloc_net, num_samples, cate_one_hot=None, real_joints=None):
        prior_net.eval()
        decoder.eval()
        veloc_net.eval()
        with torch.no_grad():
            if cate_one_hot is None:
                cate_one_hot, classes_to_generate = self.sample_z_cate(num_samples)
            else:
                classes_to_generate = cate_one_hot.max()
            prior_vec = self.tensor_fill((num_samples, self.opt.pose_dim), 0)
            prior_net.init_hidden(num_samples)
            decoder.init_hidden(num_samples)
            veloc_net.init_hidden(num_samples)

            # sample real poses from dataset
            if real_joints is None:
                # real_joints (batch_size, motion_len, 72)
                real_joints, cate_data = self.sample_real_motion_batch()

            if real_joints.shape[0] < num_samples:
                repeat_ratio = int(num_samples / real_joints.shape[0])
                real_joints = real_joints.repeat((repeat_ratio, 1, 1))
                pad_num = num_samples - real_joints.shape[0]
                if pad_num != 0:
                    real_joints = torch.cat((real_joints, real_joints[: pad_num]), dim=0)
            else:
                real_joints = real_joints[:num_samples]
            real_poses = real_joints[:, 0, :]
            real_poses = self.Tensor(real_poses.size()).copy_(real_poses)

            generate_batch = []
            for i in range(0, self.opt.motion_length):
                condition_vec = cate_one_hot
                if self.opt.time_counter:
                    time_counter = i / (self.opt.motion_length - 1)
                    time_counter_vec = self.tensor_fill((num_samples, 1), time_counter)
                    condition_vec = torch.cat((cate_one_hot, time_counter_vec), dim=1)
                # print(prior_vec.shape, condition_vec.shape)
                h = torch.cat((prior_vec, condition_vec), dim=1)

                z_t_p, mu_p, logvar_p, h_in_p = prior_net(h)

                h_mid = torch.cat((h, z_t_p), dim=1)
                lie_out, vel_mid, h_in = decoder(h_mid)
                joints_o_traj = self.pose_lie_2_joints(lie_out, real_poses, i == 0)
                if i == 0:
                    pred_joints = joints_o_traj
                else:
                    prior_traj = prior_vec[:, :3].detach()
                    prior_o_traj = prior_vec.detach() - prior_traj.repeat(1, int(prior_vec.shape[1] / 3))
                    if self.opt.use_vel_H:
                        vel_in = (prior_o_traj, joints_o_traj, vel_mid)
                    else:
                        vel_in = torch.cat((prior_o_traj, joints_o_traj, vel_mid), dim=1)
                    vel_out = veloc_net(vel_in)
                    trajec = prior_traj + vel_out
                    pred_joints = joints_o_traj + trajec.repeat(1, int(joints_o_traj.shape[1] / 3))
                prior_vec = pred_joints
                generate_batch.append(pred_joints.unsqueeze(1))
        # (batch_size, motion_len, 72)
        generate_batch = torch.cat(generate_batch, dim=1)
        return generate_batch.cpu(), classes_to_generate, None, None

    def trainIters(self, prior_net, posterior_net, decoder, veloc_net):
        self.opt_decoder = optim.Adam(decoder.parameters(), lr=0.0002, betas=(0.9, 0.999),
                                      weight_decay=0.00001)
        self.opt_prior_net = optim.Adam(prior_net.parameters(), lr=0.0002, betas=(0.9, 0.999),
                                        weight_decay=0.00001)
        self.opt_posterior_net = optim.Adam(posterior_net.parameters(), lr=0.0002, betas=(0.9, 0.999),
                                            weight_decay=0.00001)
        self.opt_veloc_net = optim.Adam(veloc_net.parameters(), lr=0.0002, betas=(0.9, 0.999),
                                        weight_decay=0.00001)

        prior_net.to(self.device)
        posterior_net.to(self.device)
        decoder.to(self.device)
        veloc_net.to(self.device)

        def save_model(file_name):
            state = {
                "prior_net": prior_net.state_dict(),
                "posterior_net": posterior_net.state_dict(),
                "decoder": decoder.state_dict(),
                "veloc_net": veloc_net.state_dict(),
                "opt_prior_net": self.opt_prior_net.state_dict(),
                "opt_posterior_net": self.opt_posterior_net.state_dict(),
                "opt_decoder": self.opt_decoder.state_dict(),
                "opt_veloc_net": self.opt_veloc_net.state_dict(),
                "iterations": iter_num
            }
            torch.save(state, os.path.join(self.opt.model_path, file_name + ".tar"))

        def load_model(file_name):
            model = torch.load(os.path.join(self.opt.model_path, file_name + '.tar'))
            prior_net.load_state_dict(model['prior_net'])
            posterior_net.load_state_dict(model['posterior_net'])
            decoder.load_state_dict(model['decoder'])
            veloc_net.load_state_dict(model['veloc_net'])

            self.opt_prior_net.load_state_dict(model['opt_prior_net'])
            self.opt_posterior_net.load_state_dict(model['opt_posterior_net'])
            self.opt_decoder.load_state_dict(model['opt_decoder'])
            self.opt_veloc_net.load_state_dict(model['opt_veloc_net'])

        if self.opt.is_continue and self.opt.isTrain:
            load_model('latest')

        iter_num = 0
        logs = OrderedDict()
        start_time = time.time()

        e_num_samples = 20
        cate_one_hot, classes = self.sample_z_cate(e_num_samples)
        np.save(os.path.join(self.opt.joints_path, "motion_class.npy"), classes)

        start_weight = self.opt.lambda_kld

        # print("Number of iterations for each epoch: %d" % (int(self.opt.iters / self.opt.batch_size)))

        while True:
            prior_net.train()
            posterior_net.train()
            decoder.train()
            veloc_net.train()

            gen_log_dict = self.train(prior_net, posterior_net, decoder, veloc_net, self.opt_prior_net, self.opt_posterior_net,
                                      self.opt_decoder, self.opt_veloc_net, self.sample_real_motion_batch)

            for k, v in gen_log_dict.items():
                if k not in logs:
                    logs[k] = [v]
                else:
                    logs[k].append(v)

            iter_num += 1

            if iter_num % self.opt.print_every == 0:
                mean_loss = OrderedDict()
                for k, v in logs.items():
                    mean_loss[k] = sum(logs[k][-1 * self.opt.print_every:]) / self.opt.print_every
                print_current_loss(start_time, iter_num, self.opt.iters, mean_loss, current_kld=self.opt.lambda_kld)

            if iter_num % self.opt.eval_every == 0:
                fake_motion, _, _, _ = self.evaluate(prior_net, decoder, veloc_net, e_num_samples, cate_one_hot)
                np.save(os.path.join(self.opt.joints_path, "motion_joints" + str(iter_num) + ".npy"), fake_motion)

            if iter_num % self.opt.save_every == 0:
                save_model(str(iter_num))

            if iter_num % self.opt.save_latest == 0:
                save_model('latest')

            if self.opt.do_kld_schedule:
                if iter_num % self.opt.update_interval == 0:
                    self.kld_weight_scheduler(iter_num, start_weight)

            if iter_num >= self.opt.iters:
                break
        return logs

