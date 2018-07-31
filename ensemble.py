from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse

import logging


from sklearn.metrics import confusion_matrix

import numpy as np
import tensorflow as tf

import i3d
from lib.dataset import ActionDataset
from lib.load_data import load_info
from lib.feed_queue import FeedQueue
from lib.label_trans import get_label_map, trans_label

_FRAME_SIZE = 224
_QUEUE_SIZE = 20
_QUEUE_PROCESS_NUM = 1
_MIX_WEIGHT_OF_RGB = 0.5
_MIX_WEIGHT_OF_FLOW = 0.5
_LOG_ROOT = 'output'

# NOTE: Before running, change the path of data
_DATA_ROOT = {
    'ucf101': {
        'rgb': '/data1/yunfeng/dataset/ucf101/jpegs_256',
        'flow': '/data1/yunfeng/dataset/ucf101/tvl1_flow/{:s}'
    },
    'hmdb51': {
        'rgb': '/data2/yunfeng/dataset/hmdb51/jpegs_256',
        'flow': '/data2/yunfeng/dataset/hmdb51/tvl1_flow/{:s}'
    },
    'clipped_data': {
        #'rgb': os.path.join('data', 'clipped_data', 'rgb'),
        #'flow': os.path.join('data', 'clipped_data', 'tvl1', 'flow-{:s}')]
        'rgb': '',
        'flow': ''
    }
}

# NOTE: Before running, change the path of checkpoints
_CHECKPOINT_PATHS = {
    #'rgb': os.path.join('output', 'finetune-clipped_data-rgb-12', 'clipped_data_rgb_0.771_model-29260'),
    #'flow': os.path.join('output', 'finetune-clipped_data-flow-12', 'clipped_data_flow_0.945_model-20900'),
    #'rgb': os.path.join('output', 'finetune-clipped_data-rgb-11', 'clipped_data_rgb_0.881_model-44450'),
    #'flow': os.path.join('output', 'finetune-clipped_data-flow-11', 'clipped_data_flow_0.929_model-31750'),
    #'rgb': os.path.join('output', 'finetune-clipped_data-rgb-10', 'clipped_data_rgb_0.943_model-46130'),
    #'flow': os.path.join('output', 'finetune-clipped_data-flow-10', 'clipped_data_flow_0.959_model-32950'),
    #'rgb': os.path.join('output', 'finetune-clipped_data-rgb-9', 'clipped_data_rgb_0.655_model-48160'),
    #'flow': os.path.join('output', 'finetune-clipped_data-flow-9', 'clipped_data_flow_0.951_model-34400'),
    #'rgb': os.path.join('output', 'finetune-clipped_data-rgb-8', 'clipped_data_rgb_0.944_model-47950'),
    #'flow': os.path.join('output', 'finetune-clipped_data-flow-8', 'clipped_data_flow_0.950_model-34250'),
    'rgb': os.path.join('output', 'finetune-clipped_data-rgb-7', 'clipped_data_rgb_0.967_model-32050'),
    'flow': os.path.join('output', 'finetune-clipped_data-flow-7', 'clipped_data_flow_0.913_model-32050'),
    #'rgb': os.path.join('output', 'finetune-clipped_data-rgb-6', 'clipped_data_rgb_0.806_model-34450'),
    #'flow': os.path.join('output', 'finetune-clipped_data-flow-6', 'clipped_data_flow_0.925_model-34450'),
    #'rgb': os.path.join('output', 'finetune-clipped_data-rgb-5', 'clipped_data_rgb_0.963_model-34050'),
    #'flow': os.path.join('output', 'finetune-clipped_data-flow-5', 'clipped_data_flow_0.973_model-34050'),
    #'rgb': os.path.join('output', 'finetune-clipped_data-rgb-4', 'clipped_data_rgb_0.986_model-33150'),
    #'flow': os.path.join('output', 'finetune-clipped_data-flow-4', 'clipped_data_flow_0.966_model-33150'),
    #'rgb': os.path.join('output', 'finetune-clipped_data-rgb-3', 'clipped_data_rgb_0.950_model-34600'),
    #'flow': os.path.join('output', 'finetune-clipped_data-flow-3', 'clipped_data_flow_0.959_model-34600'),
    #'rgb': os.path.join('output', 'finetune-clipped_data-rgb-2', 'clipped_data_rgb_0.948_model-34650'),
    #'flow': os.path.join('output', 'finetune-clipped_data-flow-2', 'clipped_data_flow_0.965_model-34650'),
    #'rgb': os.path.join('output', 'finetune-clipped_data-rgb-1', 'clipped_data_rgb_0.980_model-34350'),
    #'flow': os.path.join('output', 'finetune-clipped_data-flow-1', 'clipped_data_flow_0.953_model-34350'),
    #'rgb': '/data1/yunfeng/Lab/I3D_Finetune/model/ucf101_rgb_0.914_model-6360',
    #'flow': '/home/alexhu/I3DFORFLOW/I3D_FLOW/model/ucf101_flow_0.946_model-9540',
    #    'rgb': '/data1/yunfeng/i3d_test/model/dp_0.3_d_0.9/hmdb51_obj_rgb_0.515_model-23166',
}
_CHECKPOINT_PATHS_RGB = [
	os.path.join('output', 'finetune-clipped_data-rgb-1', 'clipped_data_rgb-34350'),
	os.path.join('output', 'finetune-clipped_data-rgb-1', 'clipped_data_rgb-27480')
]
_CHECKPOINT_PATHS_FLOW = [
	os.path.join('output', 'finetune-clipped_data-flow-1', 'clipped_data_flow-34350'),
	os.path.join('output', 'finetune-clipped_data-flow-1', 'clipped_data_flow-27480')
]

_CHANNEL = {
    'rgb': 3,
    'flow': 2,
}

_SCOPE = {
    'rgb': 'RGB',
    'flow': 'Flow',
}

_CLASS_NUM = {
    'ucf101': 101,
    'hmdb51': 51,
    'clipped_data': 8
}


def main(dataset, mode, split):
    assert mode in ['rgb', 'flow']
    log_dir = os.path.join(_LOG_ROOT, 'ensemble-%s-%s-%d' % (dataset, mode, split))
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    label_holder = tf.placeholder(tf.int32, [None])
    if mode in ['rgb']:
        rgb_holder = tf.placeholder(
            tf.float32, [None, None, _FRAME_SIZE, _FRAME_SIZE, _CHANNEL['rgb']])
    if mode in ['flow']:
        flow_holder = tf.placeholder(
            tf.float32, [None, None, _FRAME_SIZE, _FRAME_SIZE, _CHANNEL['flow']])
    #print(info_rgb)
    # insert the model
    if mode in ['rgb']:
        with tf.variable_scope(_SCOPE['rgb']):
            rgb_model = i3d.InceptionI3d(
                400, spatial_squeeze=True, final_endpoint='Logits')
            rgb_logits, _ = rgb_model(
                rgb_holder, is_training=False, dropout_keep_prob=1)
            rgb_logits_dropout = tf.nn.dropout(rgb_logits, 1)
            rgb_fc_out = tf.layers.dense(
                rgb_logits_dropout, _CLASS_NUM[dataset], tf.nn.relu, use_bias=True)
            rgb_top_1_op = tf.nn.in_top_k(rgb_fc_out, label_holder, 1)
    if mode in ['flow']:
        with tf.variable_scope(_SCOPE['flow']):
            flow_model = i3d.InceptionI3d(
                400, spatial_squeeze=True, final_endpoint='Logits')
            flow_logits, _ = flow_model(
                flow_holder, is_training=False, dropout_keep_prob=1)
            flow_logits_dropout = tf.nn.dropout(flow_logits, 1)
            flow_fc_out = tf.layers.dense(
                flow_logits_dropout, _CLASS_NUM[dataset], use_bias=True)
            flow_top_1_op = tf.nn.in_top_k(flow_fc_out, label_holder, 1)

    # construct two separate feature map and saver(rgb_saver,flow_saver)
    variable_map = {}
    if mode in ['rgb']:
        for variable in tf.global_variables():
            tmp = variable.name.split('/')
            if tmp[0] == _SCOPE['rgb']:
                variable_map[variable.name.replace(':0', '')] = variable
        rgb_saver = tf.train.Saver(var_list=variable_map)
    variable_map = {}
    if mode in ['flow']:
        for variable in tf.global_variables():
            tmp = variable.name.split('/')
            if tmp[0] == _SCOPE['flow']:
                variable_map[variable.name.replace(':0', '')] = variable
        flow_saver = tf.train.Saver(var_list=variable_map, reshape=True)

    # Edited Version by AlexHu
    if mode == 'rgb':
        fc_out = rgb_fc_out
        softmax = tf.nn.softmax(fc_out)
    if mode == 'flow':
        fc_out = flow_fc_out
        softmax = tf.nn.softmax(fc_out)
    top_k_op = tf.nn.in_top_k(softmax, label_holder, 1)

    # GPU config
    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.6
    # sess = tf.Session(config=config)# config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.6
    # sess = tf.Session(config=config)
    sess = tf.Session()
    sess1 = tf.Session()
    sess2 = tf.Session()
    
    if mode in ['rgb']:
        rgb_saver.restore(sess1, _CHECKPOINT_PATHS_RGB[0])
    if mode in ['flow']:
        flow_saver.restore(sess1, _CHECKPOINT_PATHS_FLOW[0])
    
    if mode in ['rgb']:
        rgb_saver.restore(sess2, _CHECKPOINT_PATHS_RGB[1])
    if mode in ['flow']:
        flow_saver.restore(sess2, _CHECKPOINT_PATHS_FLOW[1])
    
    
    if mode in ['rgb']:
        for variable in tf.global_variables():
            tmp = variable.name.split('/')
            if tmp[0] == _SCOPE['rgb']:
                variable = tf.devide(tf.add_n([sess1.run(variable.name), sess2.run(variable.name)]), 2)
    
    if mode in ['flow']:
        for variable in tf.global_variables():
            tmp = variable.name.split('/')
            if tmp[0] == _SCOPE['flow']:
                variable = tf.devide(tf.add_n([sess1.run(variable.name), sess2.run(variable.name)]), 2)
    
    saver = tf.Saver()
    saver.save(sess, os.path.join(log_dir, dataset+'_'+mode))
    # start a new session and restore the fine-tuned model
    '''
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    if mode in ['rgb', 'mixed']:
        rgb_saver.restore(sess, _CHECKPOINT_PATHS_RGB[int(split)-1])
    if mode in ['flow', 'mixed']:
        flow_saver.restore(sess, _CHECKPOINT_PATHS_FLOW[int(split)-1])

    if mode in ['rgb', 'mixed']:
        # Start Queue
        rgb_queue = FeedQueue(queue_size=_QUEUE_SIZE)
        rgb_queue.start_queue(rgb_data.get_video, args=info_rgb,
                              process_num=_QUEUE_PROCESS_NUM)
    if mode in ['flow', 'mixed']:
        flow_queue = FeedQueue(queue_size=_QUEUE_SIZE)
        flow_queue.start_queue(flow_data.get_video,
                               args=info_flow, process_num=_QUEUE_PROCESS_NUM)

    # Here we start the test procedure
    print('----Here we start!----')
    print('Output wirtes to '+ log_dir)
    true_count = 0
    
    if mode in ['rgb', 'mixed']:
        video_size = len(info_rgb)
    if mode in ['flow', 'mixed']:
        video_size = len(info_flow)
       
    error_record = open(os.path.join(
        log_dir, 'error_record_'+mode+'.txt'), 'w')
    rgb_fc_data = np.zeros((video_size, _CLASS_NUM[dataset]))
    flow_fc_data = np.zeros((video_size, _CLASS_NUM[dataset]))
    label_data = np.zeros((video_size, 1))

    # just load 1 video for test,this place needs to be improved
    y_pred = []
    y_true = []
    for i in range(video_size):
        #print(i)
        if mode in ['rgb', 'mixed']:
            rgb_clip, label, info = rgb_queue.feed_me()
            rgb_clip = rgb_clip/255
            #input_rgb = rgb_clip[np.newaxis, :, :, :, :]
            input_rgb = rgb_clip[np.newaxis, :, :, :, :]
            video_name = rgb_data.videos[info[0]].name
        if mode in ['flow', 'mixed']:
            flow_clip, label, info = flow_queue.feed_me()
            flow_clip = 2*(flow_clip/255)-1
            input_flow = flow_clip[np.newaxis, :, :, :, :]
            video_name = flow_data.videos[info[0]].name
        input_label = np.array([label]).reshape(-1)
        #print(type(input_label[0]))
#        print('input_rgb.shape:', input_rgb.shape)
#        print('input_flow.shape:', input_flow.shape)
#        print('input_label.shape:', input_label.shape)

        # Extract features from rgb and flow
        if mode in ['rgb']:
            top_1, predictions, curr_rgb_fc_data = sess.run(
                [top_k_op, fc_out, rgb_fc_out],
                feed_dict={rgb_holder: input_rgb,
                           label_holder: input_label})
        if mode in ['flow']:
            top_1, predictions, curr_flow_fc_data = sess.run(
                [top_k_op, fc_out, flow_fc_out],
                feed_dict={flow_holder: input_flow,
                           label_holder: input_label})
        if mode in ['mixed']:
            top_1, predictions, curr_rgb_fc_data, curr_flow_fc_data = sess.run(
                [top_k_op, fc_out, rgb_fc_out, flow_fc_out],
                feed_dict={rgb_holder: input_rgb, flow_holder: input_flow,
                           label_holder: input_label})
        if mode in ['rgb', 'mixed']:
            rgb_fc_data[i, :] = curr_rgb_fc_data
        if mode in ['flow', 'mixed']:
            flow_fc_data[i, :] = curr_flow_fc_data
        label_data[i, :] = label

        tmp = np.sum(top_1)
        true_count += tmp
        print('Video %d - frame %d-%d: %d, accuracy: %.4f (%d/%d) , name: %s' %
              (info[0], info[2], info[2]+info[1], tmp, true_count/video_size, true_count, video_size, video_name))
        logging.info('Video%d-frame%d-%d: %d, accuracy: %.4f (%d/%d) , name:%s' %
                     (info[0], info[2], info[2]+info[1], tmp, true_count/video_size, true_count, video_size, video_name))

        # self_added
#        print(predictions[0, np.argmax(predictions, axis=1)[0]])
#        print(trans_label(np.argmax(predictions, axis=1)[0], label_map))
        # print(np.argmax(label))
        #print(trans_label(np.argmax(label), label_map))
        y_true.append(trans_label(np.int64(input_label[0]), label_map))
        answer = np.argmax(predictions, axis=1)[0]
        y_pred.append(trans_label(answer, label_map))
        if tmp == 0:
            wrong_answer = np.argmax(predictions, axis=1)[0]
            #print(label_map[wrong_answer])
            # Attention: the graph output are converted into the type of numpy.array
            print('---->answer: %s, probability: %.2f' %
                  (trans_label(wrong_answer, label_map), predictions[0, wrong_answer]))
            #print(predictions)
            logging.info('---->answer: %s, probability: %.2f' %
                         (trans_label(wrong_answer, label_map), predictions[0, wrong_answer]))
            error_record.write(
                'video: %s, frame: %d-%d, answer: %s, true: %s, probability: answer-%.2f   true-%.2f\n' %
                (video_name, info[2], info[2]+info[1], trans_label(wrong_answer, label_map), trans_label(np.int64(input_label[0]), label_map),
                 predictions[0, wrong_answer], predictions[0, np.int64(input_label[0])]))
    error_record.close()
    accuracy = true_count / video_size
    print('test accuracy: %.4f' % (accuracy))
    logging.info('test accuracy: %.4f' % (accuracy))
    if mode in ['rgb', 'mixed']:
        np.save(os.path.join(log_dir, 'obj_{}_rgb_fc_{}.npy').format(
            dataset, accuracy), rgb_fc_data)
    if mode in ['flow', 'mixed']:
        np.save(os.path.join(log_dir, 'obj_{}_flow_fc_{}.npy').format(
            dataset, accuracy), flow_fc_data)
    np.save(os.path.join(log_dir, 'obj_{}_label.npy').format(dataset), label_data)

    if mode in ['rgb', 'mixed']:
        rgb_queue.close_queue()
    if mode in ['flow', 'mixed']:
        flow_queue.close_queue()
    sess.close()
    #print(y_pred)
    #print(y_true)
    cf_matrix = confusion_matrix(y_true, y_pred, labels=label_map)
    print(cf_matrix)
    np.save(os.path.join(log_dir, 'cf_matrix_{}.npy'.format(mode)), cf_matrix)
    '''


if __name__ == '__main__':
    description = 'Test Finetuned I3D Model'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('dataset', type=str, help="name of dataset, e.g., ucf101")
    p.add_argument('mode', type=str, help="type of data, e.g., rgb")
    p.add_argument('split', type=int, help="split of data, e.g., 1")
    main(**vars(p.parse_args()))
