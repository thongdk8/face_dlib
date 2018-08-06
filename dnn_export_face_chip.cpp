// The contents of this file are in the public domain. See
// LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This is an example illustrating the use of the deep learning tools from the
   dlib C++
    Library.  In it, we will show how to do face recognition.  This example uses
   the
    pretrained dlib_face_recognition_resnet_model_v1 model which is freely
   available from
    the dlib web site.  This model has a 99.38% accuracy on the standard LFW
   face
    recognition benchmark, which is comparable to other state-of-the-art methods
   for face
    recognition as of February 2017.

    In this example, we will use dlib to do face clustering.  Included in the
   examples
    folder is an image, bald_guys.jpg, which contains a bunch of photos of
   action movie
    stars Vin Diesel, The Rock, Jason Statham, and Bruce Willis.   We will use
   dlib to
    automatically find their faces in the image and then to automatically
   determine how
    many people there are (4 in this case) as well as which faces belong to each
   person.

    Finally, this example uses a network with the loss_metric loss.  Therefore,
   if you want
    to learn how to train your own models, or to get a general introduction to
   this loss
    layer, you should read the dnn_metric_learning_ex.cpp and
    dnn_metric_learning_on_images_ex.cpp examples.
*/

#include <dlib/clustering.h>
#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/string.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

using namespace dlib;
using namespace std;
using namespace cv;

// ----------------------------------------------------------------------------------------

// The next bit of code defines a ResNet network.  It's basically copied
// and pasted from the dnn_imagenet_ex.cpp example, except we replaced the loss
// layer with loss_metric and made the network somewhat smaller.  Go read the
// introductory
// dlib DNN examples to learn what all this stuff means.
//
// Also, the dnn_metric_learning_on_images_ex.cpp example shows how to train
// this network.
// The dlib_face_recognition_resnet_model_v1 model used by this example was
// trained using
// essentially the code shown in dnn_metric_learning_on_images_ex.cpp except the
// mini-batches were made larger (35x15 instead of 5x5), the iterations without
// progress
// was set to 10000, the jittering you can see below in jitter_image() was used
// during
// training, and the training dataset consisted of about 3 million images
// instead of 55.
// Also, the input layer was locked to images of size 150.
template <template <int, template <typename> class, int, typename> class block,
          int N, template <typename> class BN, typename SUBNET>
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

template <template <int, template <typename> class, int, typename> class block,
          int N, template <typename> class BN, typename SUBNET>
using residual_down =
    add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block =
    BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET>
using ares = relu<residual<block, N, affine, SUBNET>>;
template <int N, typename SUBNET>
using ares_down = relu<residual_down<block, N, affine, SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET>
using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET>
using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET>
using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<
    128,
    avg_pool_everything<alevel0<alevel1<alevel2<alevel3<alevel4<max_pool<
        3, 3, 2, 2,
        relu<affine<con<32, 7, 7, 2, 2, input_rgb_image_sized<150>>>>>>>>>>>>>;

// ----------------------------------------------------------------------------------------

template <long num_filters, typename SUBNET>
using con5d = con<num_filters, 5, 5, 2, 2, SUBNET>;
template <long num_filters, typename SUBNET>
using con5 = con<num_filters, 5, 5, 1, 1, SUBNET>;

template <typename SUBNET>
using downsampler = relu<affine<
    con5d<32, relu<affine<con5d<32, relu<affine<con5d<16, SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon5 = relu<affine<con5<45, SUBNET>>>;

using net_type = loss_mmod<
    con<1, 9, 9, 1, 1, rcon5<rcon5<rcon5<downsampler<
                           input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

// ----------------------------------------------------------------------------------------

// ----------------------------------------------------------------------------------------

std::vector<matrix<rgb_pixel>> jitter_image(const matrix<rgb_pixel> &img);

// ----------------------------------------------------------------------------------------

int main(int argc, char **argv) try {
    if (argc < 2) {
        cout << "Run this example by invoking it like this: " << endl;
        cout << "   ./dnn_face_recognition_ex path_to_video path_to_stored_facechip" << endl;
        cout << endl;
        return 1;
    }

    // The first thing we are going to do is load all our models.  First, since
    // we need to
    // find faces in the image we will need a face detector:
    // frontal_face_detector detector = get_frontal_face_detector();

    net_type detector;
    deserialize("/home/thongpb/works/face_recognition/dlib-models-master/"
                "mmod_human_face_detector.dat") >>
        detector;

    // We will also use a face landmarking model to align faces to a standard
    // pose:  (see face_landmark_detection_ex.cpp for an introduction)
    shape_predictor sp;
    deserialize("/home/thongpb/works/face_recognition/dlib-models-master/"
                "shape_predictor_68_face_landmarks.dat") >>
        sp;
    // And finally we load the DNN responsible for face recognition.
    anet_type net;
    deserialize("/home/thongpb/works/face_recognition/dlib_implementation/"
                "build/metric_network_renset.dat") >>
        net;

    std::cout << "Load models successfully" << endl;

    cv::VideoCapture vi_cap = cv::VideoCapture(argv[1]);

    if (!vi_cap.isOpened())
        return -1;

    cv::Mat frame =
        imread("/home/thongpb/works/face_recognition/data/duong3.png");
    int count_face = 0;
    while (true) {

        vi_cap.read(frame);
        if (frame.empty())
            break;

        // cv::resize(frame, frame, Size(960,540));

        dlib::cv_image<rgb_pixel> frame_dlib(frame);
        dlib::matrix<rgb_pixel> img;
        dlib::assign_image(img, frame_dlib);

        // matrix<rgb_pixel> img;
        // load_image(img, argv[1]);
        // Display the raw image on the screen
        // image_window win(img);

        // Run the face detector on the image of our action heroes, and for each
        // face extract a
        // copy that has been normalized to 150x150 pixels in size and
        // appropriately
        // rotated
        // and centered.
        
        std::vector<matrix<rgb_pixel>> faces;
        auto dets = detector(img);
        for (auto face : dets) {
            auto shape = sp(img, face.rect);
            matrix<rgb_pixel> face_chip;
            extract_image_chip(img, get_face_chip_details(shape, 150, 0.25),
                               face_chip);
            faces.push_back(face_chip);

            cv::Mat face_chip_cv = dlib::toMat(face_chip);
            cv::imshow("face", face_chip_cv);
            cv::imwrite(argv[2]+std::to_string(count_face) + ".png", face_chip_cv);

            long left_cor = face.rect.left();
            long right_cor = face.rect.right();
            long top_cor = face.rect.top();
            long bottom_cor = face.rect.bottom();
            cv::rectangle(frame, cv::Point(left_cor, top_cor),
                          cv::Point(right_cor, bottom_cor),
                          cv::Scalar(255, 120, 120), 2);
            
            count_face ++;
        }

        if (faces.size() == 0) {
            cout << "No faces found in image!" << endl;
            //break;
            // return 1;
        }

        // This call asks the DNN to convert each face image in faces into a
        // 128D
        // vector.
        // In this 128D vector space, images from the same person will be close
        // to
        // each other
        // but vectors from different people will be far apart.  So we can use
        // these
        // vectors to
        // identify if a pair of images are from the same person or from
        // different
        // people.
        else{
            // std::vector<matrix<float, 0, 1>> face_descriptors = net(faces);
            // cout << "face descriptor for one face: "
            //      << trans(face_descriptors[0]) << endl;
        }
        

        cv::imshow("frame", frame);
        int keyy = cv::waitKey(1);
        if (keyy == 27)
            break;
    }

} catch (std::exception &e) {
    cout << e.what() << endl;
}

// ----------------------------------------------------------------------------------------

std::vector<matrix<rgb_pixel>> jitter_image(const matrix<rgb_pixel> &img) {
    // All this function does is make 100 copies of img, all slightly jittered
    // by being
    // zoomed, rotated, and translated a little bit differently.
    thread_local random_cropper cropper;
    cropper.set_chip_dims(150, 150);
    cropper.set_randomly_flip(true);
    cropper.set_max_object_size(0.99999);
    cropper.set_background_crops_fraction(0);
    cropper.set_min_object_size(0.97);
    cropper.set_translate_amount(0.02);
    cropper.set_max_rotation_degrees(3);

    std::vector<mmod_rect> raw_boxes(1), ignored_crop_boxes;
    raw_boxes[0] = shrink_rect(get_rect(img), 3);
    std::vector<matrix<rgb_pixel>> crops;

    matrix<rgb_pixel> temp;
    for (int i = 0; i < 100; ++i) {
        cropper(img, raw_boxes, temp, ignored_crop_boxes);
        crops.push_back(move(temp));
    }
    return crops;
}

// ----------------------------------------------------------------------------------------
