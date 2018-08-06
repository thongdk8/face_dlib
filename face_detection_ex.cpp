// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This example program shows how to find frontal human faces in an image.  In
    particular, this program shows how you can take a list of images from the
    command line and display each on the screen with red boxes overlaid on each
    human face.

    The examples/faces folder contains some jpg images of people.  You can run
    this program on them and see the detections by executing the following command:
        ./face_detection_ex faces/*.jpg

    
    This face detector is made using the now classic Histogram of Oriented
    Gradients (HOG) feature combined with a linear classifier, an image pyramid,
    and sliding window detection scheme.  This type of object detector is fairly
    general and capable of detecting many types of semi-rigid objects in
    addition to human faces.  Therefore, if you are interested in making your
    own object detectors then read the fhog_object_detector_ex.cpp example
    program.  It shows how to use the machine learning tools which were used to
    create dlib's face detector. 


    Finally, note that the face detector is fastest when compiled with at least
    SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
    chip then you should enable at least SSE2 instructions.  If you are using
    cmake to compile this program you can enable them by using one of the
    following commands when you create the build project:
        cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
    This will set the appropriate compiler options for GCC, clang, Visual
    Studio, or the Intel compiler.  If you are using another compiler then you
    need to consult your compiler's manual to determine how to enable these
    instructions.  Note that AVX is the fastest but requires a CPU from at least
    2011.  SSE4 is the next fastest and is supported by most current machines.  
*/

#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv/cv_image.h>
#include <iostream>


#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

using namespace dlib;
using namespace std;


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


// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{  
    try
    {
        if (argc == 1)
        {
            cout << "Give some image files as arguments to this program." << endl;
            return 0;
        }

        frontal_face_detector detector = get_frontal_face_detector();

        cv::VideoCapture vi_cap = cv::VideoCapture(argv[1]);

        if (!vi_cap.isOpened())
            return -1;

        cv::Mat frame;

        ChronoTimer tm;


        while (true) {
            
            vi_cap.read(frame);

            if (frame.empty())
                break;

            cv::Mat org_fr = frame.clone();
            cv::imshow("original_frame", org_fr);

            // cv::resize(frame, frame, cv::Size(0, 0),0.5,0.5);
            cv::pyrDown(frame, frame, cv::Size(frame.cols/2, frame.rows/2));

            // cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

            dlib::cv_image<rgb_pixel> frame_dlib(frame);
            dlib::matrix<rgb_pixel> frame_mat_dlib;
            dlib::assign_image(frame_mat_dlib, frame_dlib);

            // while (frame_mat_dlib.size() < 1920 * 1080)
            //     pyramid_up(frame_mat_dlib);

            tm.start();
            std::vector<rectangle> dets = detector(frame_mat_dlib);

            tm.stop();
            cout << tm.HZ() << endl;
            int num_faces = 0;
            for (auto d : dets) {
                long left_cor = d.left();
                long right_cor = d.right();
                long top_cor = d.top();
                long bottom_cor = d.bottom();

                cv::rectangle(frame, cv::Point(left_cor, top_cor),
                              cv::Point(right_cor, bottom_cor),
                              cv::Scalar(255, 120, 120), 2);
                num_faces++;
            }

            cv::imshow("detected_frame", frame);
            int key = cv::waitKey(1);
            if (key == 27)
                break;
        }

    }
    catch (exception& e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }
}

// ----------------------------------------------------------------------------------------

