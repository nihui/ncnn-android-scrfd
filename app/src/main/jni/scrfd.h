// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef SCRFD_H
#define SCRFD_H

#include <opencv2/core/core.hpp>

#include <net.h>

struct FaceObject
{
    cv::Rect_<float> rect;
    cv::Point2f landmark[5];
    float prob;
};

class SCRFD
{
public:
    int load(const char* modeltype, bool use_gpu = false);

    int load(AAssetManager* mgr, const char* modeltype, bool use_gpu = false);

    int detect(const cv::Mat& rgb, std::vector<FaceObject>& faceobjects, float prob_threshold = 0.5f, float nms_threshold = 0.45f);

    int draw(cv::Mat& rgb, const std::vector<FaceObject>& faceobjects);

private:
    ncnn::Net scrfd;
    bool has_kps;
};

#endif // SCRFD_H
