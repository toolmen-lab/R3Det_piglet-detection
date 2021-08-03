import matplotlib.pyplot as plt
import numpy as np
import sys
import json
import pickle
sys.path.append("../libs")
from configs import cfgs

def plot_roc_curve():
    classes_names = get_categories()
#    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, constrained_layout=True)
#    fig.set_size_inches(16, 6)
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(111)
    recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
    acc = []
    lines = []
    aps = []
    for i in range(3):
        with open('../output/predictions/RetinaNet_PIGLET_20200603/test{}/eval_parameters.pkl'.format(i+1), 'rb') as file:
            parameters = pickle.load(file)
            precisions = parameters["precision"]
            recalls = parameters["recall"]
            
            # PR curve with different categories
            
            f_scores = np.linspace(0.2, 0.8, num=4)
            for f_score in f_scores:
                x = np.linspace(0.01, 1)
                y = f_score * x / (2 * x - f_score)
                f1_curve, = ax1.plot(x[y >= 0], y[y >= 0], color=[192/255 for _ in range(3)], alpha = 0.3, linestyle = "--", label = "iso-f1 curves")
                ax1.annotate('F1={0:0.1f}'.format(f_score), xy=(x[np.where(np.floor(y) == 1)[0][-1]], 1), fontsize = 10)
            for idx in range(precisions.shape[2]):
                precision = precisions[0, :, idx, 0, 2]
                precision = precision[precision > -1]
                ap = np.mean(precision) if precision.size else float("nan")
                print(ap)
                aps.append(ap)
                line, = ax1.plot(recThrs, precision, label = "Camera {} (AP={:.4f})".format(i+1, ap))
            lines.append(line)
            p, r, f1 = caluculate_f1_score(classes_names, precisions, recalls)
#            ax1.plot(r, p, color = line.get_color(), marker="o")
            acc.append([p, r])

    acc = np.array(acc)
    mean_p, mean_r = np.mean(acc, axis=0)
    mean_F1 = 2 * mean_p * mean_r / (mean_r + mean_p)
    print("Overall evaluation: Precision={:.4f}, Recall={:.4f}, F1-score={:.4f}".format(mean_p, mean_r, mean_F1))
    print("Overall mAP {}".format(sum(aps)/3))
    
    ax1.legend(handles = [f1_curve, lines[0], lines[1], lines[2]], loc = "lower left", frameon=True)
#    ax1.set_title('PR curve with different categories at iou 0.5 (mAP={:.3f})'.format(mean_p))
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_ylim([-0.05, np.amax(precision) + 0.05])
    ax1.set_aspect(1, adjustable = 'box')
    ax1.grid(linestyle="--", alpha=0.3)
#        # Recalls with different IoU threshold (100 proposals)
#        for idx in range(recalls.shape[2]):
#            recall = recalls[:, idx, 0, 1]
#            recall = recall[recall > -1]
#            ax2.plot(iouThrs, recall, label = classes_names[idx], color = [item/255 for item in Color[idx]], marker = '^')
#        ax2.legend(loc = 'lower left', title = 'Categories')
#        ax2.set_title('Recalls with different IoU threshold (10 proposals)')
#        ax2.set_xlabel('IoU Threshold')
#        ax2.set_xlim([0.45, 1])
#        ax2.set_ylabel('Recall')
#        ax2.set_aspect(0.5, adjustable = 'box')
#        
#        # PR curve with different IoU threshold
#        aps = []
#        colors = plt.cm.jet(np.linspace(0, 0.9, precisions.shape[0]))
#        for i in range(precisions.shape[0]):
#            precision = np.mean(precisions[i, :, :, 0, -1], axis=1)
#            precision = precision[precision > -1]
#            ap = np.mean(precision) if precision.size else float("nan")
#            ax3.plot(recThrs, precision, label = '{:.2f} (AP={:.2f})'.format(0.5+0.05*i, ap), color = colors[i])
#
#        ax3.legend(loc = 'upper right', title = 'IoU Thr', fontsize = 8)
#        ax3.set_title('PR curve with different IoU threshold')
#        ax3.set_xlabel('Recall')
#        ax3.set_ylabel('Precision')
#        ax3.set_aspect(1, adjustable = 'box')       
#       
#        fig.suptitle("NMS IOU THRESHOLD at {}".format(cfgs.NMS_IOU_THRESHOLD), fontsize = 16)
#
#def plot_f1_score():
#    classes_names = get_categories()
#    with open('../output/predictions/RetinaNet_PIGLET_20200414/eval_parameters.pkl', 'rb') as file:
#        parameters = pickle.load(file)
#        precisions = parameters["precision"]
#        recalls = parameters["recall"]
#        
#        fig = plt.figure()
#        ax = fig.add_subplot(111)
#        colors = plt.cm.jet(np.linspace(0, 1, recalls.shape[0]))
#        print("-" * 100)
#        for t in range(recalls.shape[0]):
#            iou_thresh = 0.5 + 0.05 * t
#            for idx in classes_names.keys():
#                precision = precisions[t, :, idx, 0, 1]
#                precision = precision[precision > 0]
#                p = precision[-1] if precision.size else 0
#                r = recalls[t, idx, 0, 1]
#                F1 = 2 * p * r / (r + p)
#                F1 = 0 if np.isnan(F1) else F1
#                print("Class {} at IoU {:.2f}, F1-score:{:.3f}, Precision:{:.3f}, Recall:{:.3f} ".format(classes_names[idx], iou_thresh, F1, p, r))
#            p = np.mean(precisions[t, :, :, 0, 1])
#            r = np.mean(recalls[t, :, 0, 1])
#            F1 = 2 * p * r / (r + p)
#            F1 = 0 if np.isnan(F1) else F1
#            ax.scatter(p, r, label = "F1-score:{:.2f}, IoU:{:.2f}".format(F1, iou_thresh), color = colors[t])
#            print("F1-score:{:.3f} at IoU {:.2f}".format(F1, iou_thresh))
#            print("-" * 100)
#        ax.set_xlabel("Precision")
#        ax.set_ylabel("Recall")
#        ax.set_xlim([-0.05, 1.05])
#        ax.set_ylim([-0.05, 1.05])
#        ax.set_title("F1-score under different IoU")
#        ax.legend()
#        p = np.mean(precisions[:, :, :, 0, 1])
#        r = np.mean(recalls[:, :, 0, 1])
#        F1 = 2 * p * r / (r + p)
#        print("Overall F1-score:{:.3f}".format(F1))
        
        
def caluculate_f1_score(classes_names, precisions, recalls):
    for idx in classes_names.keys():
        precision = precisions[0, :, idx, 0, 2]
        precision = precision[precision > 0]
        p = precision[-1] if precision.size else 0
        r = recalls[0, idx, 0, 2]
        F1 = 2 * p * r / (r + p)
        F1 = 0 if np.isnan(F1) else F1
        print("Class {} at IoU 0.5 | F1-score:{:.3f} | Precision:{:.3f} | Recall:{:.3f} ".format(classes_names[idx], F1, p, r))
    return p, r, F1
    
def plot_loss():
    with open("../output/losses/run-.-tag-cls_cls_loss.json") as f:
        cls_loss = json.load(f)
        cls_loss = smooth([loss[2] for loss in cls_loss])
    
    with open("../output/losses/run-.-tag-refine_refine_cls_loss.json") as f:
        refine_cls_loss = json.load(f)
        refine_cls_loss = smooth([loss[2] for loss in refine_cls_loss])      
    
    with open("../output/losses/run-.-tag-refine_refine_reg_loss.json") as f:
        refine_reg_loss = json.load(f)
        refine_reg_loss = smooth([loss[2] for loss in refine_reg_loss])
        
    with open("../output/losses/run-.-tag-reg_reg_loss.json") as f:
        reg_loss = json.load(f)
        reg_loss = smooth([loss[2] for loss in reg_loss])
    
    with open("../output/losses/run-.-tag-total_total_losses.json") as f:
        total_loss = json.load(f)
        total_loss = smooth([loss[2] for loss in total_loss])
        
#    with open("../output/losses/run-.-tag-lr.json") as f:
#        lr = json.load(f)
#        lr = [item[2] for item in lr]
#    iteration = np.arange(0, cfgs.MAX_ITERATION + cfgs.SMRY_ITER, cfgs.SMRY_ITER)
    iteration = [i * cfgs.SMRY_ITER for i in range(len(total_loss))]
        
    
    # plot data
    fig = plt.figure(constrained_layout=False, figsize=(10,6))
    fig.suptitle('Training history', fontsize=16)
    gs = fig.add_gridspec(nrows=4, ncols=4, wspace=0.2, hspace=0.3)
    
    # plot total loss
    ax1 = fig.add_subplot(gs[:, 0:3])
    l1 = ax1.plot(iteration, total_loss, color="blue", linewidth=2.0, label=r'$L_{total}$')
    l2 = ax1.plot(iteration, cls_loss, color="green", linewidth=2.0, label=r'$L_{cls1}$')
    l3 = ax1.plot(iteration, refine_cls_loss, color="purple", linewidth=2.0, label=r'$L_{cls2}$')
    l4 = ax1.plot(iteration, reg_loss, color="red", linewidth=2.0, label=r'$L_{box1}$')
    l5 = ax1.plot(iteration, refine_reg_loss, color="orange", linewidth=2.0, label=r'$L_{box2}$')
#    ax_lr = ax1.twinx()
#    l6 = ax_lr.plot(iteration, lr, color="gray", linewidth=1.0, label=r'Learning rate', linestyle="--", alpha=0.3)
#    ax_lr.set_yticks([])
#    ax_lr.grid(False)
#    ax_lr.set_ylabel("lr")
    lines = l1+l2+l3+l4+l5
    labels = [l.get_label() for l in lines]
    ax1.set_xlabel('Iterations', fontsize=16)
    ax1.set_ylabel('Loss', fontsize=16)
    ax1.set_ylim([-0.05, 2])
#    ax1.set_xlim([0, max(iteration)])
    ax1.legend(lines, labels, loc = 0, fontsize=16, frameon=True)
    ax1.grid(linestyle="--", alpha=0.3)
#    ax1.set_title('Total loss')
        
#    # plot R3Det cls loss
#    ax2 = fig.add_subplot(gs[0, 3])
#    ax2.plot(iteration, cls_loss, color="green", linewidth=1.2)
#    ax2.set_title(r'$L_{cls1}$')
#    ax2.yaxis.set_ticks_position('right')
#    ax2.set_xticklabels([])
#    ax2.grid(linestyle=':')
#    ax2.set_xticks([0, iteration[-1]/2 , iteration[-1]])
#    
#    # plot R3Det refine cls loss
#    ax3 = fig.add_subplot(gs[1, 3])
#    ax3.plot(iteration, refine_cls_loss, color="purple", linewidth=1.2)
#    ax3.yaxis.set_ticks_position('right')
#    ax3.set_xticklabels([])
#    ax3.set_title(r'$L_{cls2}$')
#    ax3.grid(linestyle=':')
#    ax3.set_xticks([0, iteration[-1]/2 , iteration[-1]])
#    
#    # plot R3Det box loss
#    ax4 = fig.add_subplot(gs[2, 3])
#    ax4.plot(iteration, reg_loss, color="red", linewidth=1.2)
#    ax4.yaxis.set_ticks_position('right')
#    ax4.set_xticklabels([])
#    ax4.set_title(r'$L_{box1}$')
#    ax4.grid(linestyle=':')
#    ax4.set_xticks([0, iteration[-1]/2 , iteration[-1]])
#    
#    # plot R3Det refine box loss
#    ax5 = fig.add_subplot(gs[3, 3])
#    ax5.plot(iteration, refine_reg_loss, color="orange", linewidth=1.2)
#    ax5.yaxis.set_ticks_position('right')
#    ax5.set_title(r'$L_{box2}$')
#    ax5.grid(linestyle=':')
#    ax5.set_xlabel('Iterations')
#    ax5.set_xticks([0, iteration[-1]/2 , iteration[-1]])
               
    
def get_categories():
    with open('classes.txt') as f:
        classes = f.read()
    return {i: name for i, name in enumerate(classes.split(','))}
                        
def smooth(data, weight = 0.6):
    last = data[0]
    smoothed = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

            
if __name__ == '__main__':
#    plt.style.use("seaborn")
    plot_loss()
    plot_roc_curve()
#    plot_f1_score()
    plt.show()
#    plt.style.use('default')