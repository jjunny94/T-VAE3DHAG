import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, writers
import matplotlib.animation as anim
import mpl_toolkits.mplot3d.axes3d as p3
import utils.paramUtil as paramUtil
import PyQt5
import pickle as pkl
import os
from pycocotools.coco import COCO
# import joblib
# seq_name = 'downtown_walking_00'
# datasetDir = '/home/deep/action-to-motion/dataset/3dpw/sequenceFiles'
# file = os.path.join(datasetDir,'sequenceFiles/test',seq_name+'.pkl')
# #print(type(file))
# with open(file, 'rb') as f:
#     u = pkl._Unpickler(f)
#     u.encoding = 'latin1'
#     seq = u.load()
paused = False
data = np.load("/home/deep/action-to-motion/dataset/humanact12/P02G01R03F1126T1221A0102.npy", allow_pickle=True)
# data = np.load("/home/deep/Pose_to_SMPL/custom/ko8s1A1301.npy", allow_pickle=True)
# data = np.load("/home/deep/human_body_prior/support_data/dowloads/amass_sample.npz", allow_pickle=True)
#dist = data[:, :, 2]
data[:, :, 1] = data[:, :, 1] - 1
data[:, :, 0] = data[:, :, 0] + 0.2
# data[:, :, 0] = (data[:, :, 0] - np.min(data[:, :, 0]))  / (np.max(data[:, :, 0]) - np.min(data[:, :, 0]))
# data[:, :, 1] = (data[:, :, 1] - np.min(data[:, :, 1]))  / (np.max(data[:, :, 1]) - np.min(data[:, :, 1]))
# data[:, :, 2] = (data[:, :, 2] - np.min(data[:, :, 2]))  / (np.max(data[:, :, 2]) - np.min(data[:, :, 2]))


# data[:, :, :]
# db = COCO('/home/deep/Downloads/3DPW_train_new.json')
kinematic_chain = paramUtil.humanact12_kinematic_chain


def draw(motion, kinematic_tree, save_path, interval=60, dataset=None):
    matplotlib.use('Qt5Agg')
    # def toggle_pause(self):
        # global paused
        # if paused:
        #     ani.resume()
        # else:
        #     ani.pause()
        # paused = not paused

    def init():
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        if dataset == "mocap":
            ax.set_ylim(-1.5, 1.5)
            ax.set_xlim(0, 3)
            ax.set_zlim(-1.5, 1.5)
        else:
            ax.set_ylim3d(-1, 1)
            ax.set_xlim3d(-1, 1)
            ax.set_zlim3d(-1, 1)
            # ax.set_zlim(-1, 1)np.max(dist), 1000, np.min(dist)

    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax = p3.Axes3D(fig)
    init_bool = True
    init()
    # if init_bool:
        
    #     init_bool = False
    # fig.canvas.mpl_connect('button_press_event', toggle_pause)
    data = np.array(motion, dtype=float)
    colors = ['red', 'magenta', 'black', 'green', 'blue']
    frame_number = data.shape[0]
    #print(frame_number)
    # dim (frame, joints, xyz)
    print(data.shape)

    def update(index):
        ax.lines = []
        ax.collections = []
        if dataset == "mocap":
            ax.view_init(elev=110, azim=-90)
        else:
            ax.view_init(elev=110, azim=90)
        for chain, color in zip(kinematic_tree, colors):
            ax.plot3D(motion[index, chain, 0], motion[index, chain, 1], motion[index, chain, 2], linewidth=4.0, color=color)
        # plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        # ax.set_zticklabels([np.min(dist)], 1000, np.max(dist), fontsize=12)
    
    ani = FuncAnimation(fig, update, frames=frame_number, interval=interval, repeat=False, repeat_delay=200)
    # update(1)
    # plt.show()
    # Writer = writers['ffmpeg']
    # writer = Writer(fps=15, metadata={})

    plt.show()
    ani.save(save_path, writer='pillow')
    # plt.close()
# offset_mat = np.tile(data[0, 0], (data.shape[1], 1))
# data = data - offset_mat
draw(data, kinematic_chain, './P02G01R03F1126T1221A0102.gif')
