from pycocotools.cocoeval import COCOeval
from libs.box_utils import iou_rotate
import numpy as np


class RotatedCOCOeval(COCOeval):
    @staticmethod
    def check(boxlist, output_box_dim):
        if type(boxlist) == list:
            if boxlist == []:
                return np.zeros((0, output_box_dim), dtype=np.float32)
            else:
                box_tensor = np.array(boxlist)
        else:
            raise Exception("Unrecognized boxlist type")

        return box_tensor

    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return []
        inds = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0 : p.maxDets[-1]]

        assert p.iouType == "bbox", "unsupported iouType for iou computation"

        g = [g["bbox"] for g in gt]
        d = [d["bbox"] for d in dt]

        # compute iou between each dt and gt region
        iscrowd = [int(o["iscrowd"]) for o in gt]

        # Note: this function is copied from cocoeval.py in cocoapi
        # and the major difference is here.
        ious = self.compute_iou_dt_gt(d, g, iscrowd)
        return ious

    def compute_iou_dt_gt(self, dt, gt, is_crowd):
        # TODO: take is_crowd into consideration
        assert all(c == 0 for c in is_crowd)

        dt = self.check(dt, output_box_dim = 5)
        gt = self.check(gt, output_box_dim = 5)
        return iou_rotate.iou_rotate_calculate1(np.array(dt), np.array(gt), use_gpu=False)





def evaluate_predictions_on_coco(coco_gt, coco_results):
    """
    Evaluate the coco results using COCOEval API.
    """
    assert len(coco_results) > 0

    coco_dt = coco_gt.loadRes(coco_results)

    # Only bbox is supported for now
    coco_eval = RotatedCOCOeval(coco_gt, coco_dt, iouType="bbox")

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval
