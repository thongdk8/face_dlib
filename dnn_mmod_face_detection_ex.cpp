
#include <iostream>
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/opencv/cv_image.h>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace dlib;
using namespace cv;
// ----------------------------------------------------------------------------------------

template <long num_filters, typename SUBNET> using con5d = con<num_filters,5,5,2,2,SUBNET>;
template <long num_filters, typename SUBNET> using con5  = con<num_filters,5,5,1,1,SUBNET>;

template <typename SUBNET> using downsampler  = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16,SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon5  = relu<affine<con5<45,SUBNET>>>;

using net_type = loss_mmod<con<1,9,9,1,1,rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

// ----------------------------------------------------------------------------------------

class ChronoTimer {
  public:
    std::chrono::steady_clock::time_point _start;
    std::chrono::steady_clock::time_point _end;
    unsigned long _count;

    ChronoTimer() { _count = 0; }

    void start() { _start = std::chrono::steady_clock::now(); }

    void stop() { _end = std::chrono::steady_clock::now(); }

    void update() { _count += 1; }

    double elapsed() {
        return std::chrono::duration<double>(_end - _start).count();
    }

    double HZ() {
        return (double)1.0 /
               (std::chrono::duration<double>(_end - _start).count());
    }
};

int main(int argc, char** argv) try
{
    ChronoTimer tm;
    if (argc == 1)
    {
        cout << "Call this program like this:" << endl;
        cout << "./dnn_mmod_face_detection_ex mmod_human_face_detector.dat path_to_video" << endl;
        cout << "\nYou can get the mmod_human_face_detector.dat file from:\n";
        cout << "http://dlib.net/files/mmod_human_face_detector.dat.bz2" << endl;
        return 0;
    }

    net_type net;
    deserialize(argv[1]) >> net;

    cv::VideoCapture vi_cap = cv::VideoCapture(argv[2]);

    if(!vi_cap.isOpened())
        return -1;

    cv::Mat frame;

    while(true)
    {
        tm.start();
        vi_cap.read(frame);

        if(frame.empty())
            break;

        cv::Mat org_fr = frame.clone();
        // cv::imshow("original_frame", org_fr);

        // cv::resize(frame, frame, cv::Size(0, 0),0.5,0.5);
        // cv::pyrDown(frame, frame, cv::Size(frame.cols/2, frame.rows/2));

        // cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

        dlib::cv_image<rgb_pixel> frame_dlib(frame);
        dlib::matrix<rgb_pixel> frame_mat_dlib;
        dlib::assign_image(frame_mat_dlib, frame_dlib);

        // while (frame_mat_dlib.size() < 1920 * 1080)
        //     pyramid_up(frame_mat_dlib);

        auto dets = net(frame_mat_dlib);
        int num_faces = 0;
        for (auto d : dets){
            long left_cor = d.rect.left();
            long right_cor = d.rect.right();
            long top_cor = d.rect.top();
            long bottom_cor = d.rect.bottom();

            //std::cout << "Face["<< num_faces << "] whith confidence: " << d.detection_confidence << endl;

            cv::rectangle(frame, cv::Point(left_cor, top_cor),
                          cv::Point(right_cor, bottom_cor),
                          cv::Scalar(255, 120, 120), 2);
            num_faces ++;
        }

        cv::imshow("detected_frame", frame);
        tm.stop();
        cout << "FPS: " << tm.HZ() << endl;
        int key = cv::waitKey(1);
        if(key == 27)
            break;

    }

}
catch(std::exception& e)
{
    cout << e.what() << endl;
}


