/**
 * @file demo_sphereview_data.cpp
 * @brief Generating training data for CNN with triplet loss.
 * @author Yida Wang
 */
#include <opencv2/cnn_3dobj.hpp>
#include <opencv2/viz/vizcore.hpp>
#include <iostream>
#include <stdlib.h>
#include <time.h>
using namespace cv;
using namespace std;
using namespace cv::cnn_3dobj;

/**
 * @function listDir
 * @brief Making all files names under a directory into a list
 */
static void listDir(const char *path, std::vector<String>& files, bool r)
{
    DIR *pDir;
    struct dirent *ent;
    char childpath[512];
    pDir = opendir(path);
    memset(childpath, 0, sizeof(childpath));
    while ((ent = readdir(pDir)) != NULL)
    {
        if (ent->d_type & DT_DIR)
        {
            if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0 || strcmp(ent->d_name, ".DS_Store") == 0)
            {
                continue;
            }
            if (r)
            {
                sprintf(childpath, "%s/%s", path, ent->d_name);
                listDir(childpath,files,false);
            }
        }
        else
        {
            if (strcmp(ent->d_name, ".DS_Store") != 0)
                files.push_back(ent->d_name);
        }
    }
    sort(files.begin(),files.end());
}

int main(int argc, char *argv[])
{
    const String keys = "{help | | demo :$ ./sphereview_test -ite_depth=2 -plymodel=../data/3Dmodel/ape.ply -imagedir=../data/images_all/ -labeldir=../data/label_all.txt -num_class=6 -label_class=0, then press 'q' to run the demo for images generation when you see the gray background and a coordinate.}"
    "{ite_depth | 3 | Iteration of sphere generation.}"
    "{plymodel | ../data/3Dmodel/ape.ply | Path of the '.ply' file for image rendering. }"
    "{imagedir | ../data/images_all/ | Path of the generated images for one particular .ply model. }"
    "{labeldir | ../data/label_all.txt | Path of the generated images for one particular .ply model. }"
    "{semisphere | 1 | Camera only has positions on half of the whole sphere. }"
    "{z_range | 0.6 | Maximum camera position on z axis. }"
    "{center_gen | 0 | Find center from all points. }"
    "{image_size | 128 | Size of captured images. }"
    "{label_class |  | Class label of current .ply model. }"
    "{label_item |  | Item label of current .ply model. }"
    "{rgb_use | 0 | Use RGB image or grayscale. }"
    "{num_class | 6 | Total number of classes of models. }"
    "{binary_out | 0 | Produce binaryfiles for images and label. }"
    "{view_region | 0 | Take a special view of front or back angle}";
    /* Get parameters from comand line. */
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Generating training data for CNN with triplet loss");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    int ite_depth = parser.get<int>("ite_depth");
    String plymodel = parser.get<String>("plymodel");
    String imagedir = parser.get<String>("imagedir");
    
    String labeldir = parser.get<String>("labeldir"); // *
    
    int label_class = parser.get<int>("label_class");
    int label_item = parser.get<int>("label_item");
    int semisphere = parser.get<int>("semisphere");
    float z_range = parser.get<float>("z_range");
    int center_gen = parser.get<int>("center_gen");
    int image_size = parser.get<int>("image_size");
    int rgb_use = parser.get<int>("rgb_use");
    int num_class = parser.get<int>("num_class");
    int binary_out = parser.get<int>("binary_out");
    
    double obj_dist, bg_dist, y_range;
    
    
    
    std::fstream imglabel;
    imglabel.open(labeldir.c_str(), fstream::app|fstream::out);
    
    cv::cnn_3dobj::icoSphere ViewSphere( 10,2 );
    
    if (binary_out)
    {
        ViewSphere.createHeader(1, image_size, image_size, headerPath);
    }


    
    /* Images will be saved as .png files. */
    size_t cnt_img;
    srand((int)time(0));
    //std::vector<cv::Point3d> campos; -> poses
    do
    {
        cnt_img = 0;
        for(int pose = 0; pose < static_cast<int>(campos.size()); pose++){
            /* Add light. */
//             double alpha1 = rand()%(314/2)/100;
//             double alpha2 = rand()%(314*2)/100;
//             printf("%f %f %f/n", ceil(10000*sqrt(1 - sin(alpha1)*sin(alpha1))*sin(alpha2)), 10000*sqrt(1 - sin(alpha1)*sin(alpha1))*cos(alpha2), sin(alpha1)*10000);
//             myWindow.addLight(Vec3d(10000*sqrt(1 - sin(alpha1)*sin(alpha1))*sin(alpha2),10000*sqrt(1 - sin(alpha1)*sin(alpha1))*cos(alpha2),sin(alpha1)*10000), Vec3d(0,0,0), viz::Color::white(), viz::Color::white(), viz::Color::black(), viz::Color::white());
            int label_x, label_y, label_z;
            label_x = static_cast<int>(campos.at(pose).x*100);
            label_y = static_cast<int>(campos.at(pose).y*100);
            label_z = static_cast<int>(campos.at(pose).z*100);
            sprintf (temp,"%02i_%02i_%04i_%04i_%04i_%02i", label_class, label_item, label_x, label_y, label_z, static_cast<int>(obj_dist/100));
            String filename = temp;
            filename += ".png";
            imglabel << filename << ' ' << label_class << endl;
            filename = imagedir + filename;
            /* Get the pose of the camera using makeCameraPoses. */
            if (view_region != 0)
            {
                cam_focal_point.x = cam_focal_point.y - label_x/5;
            }
            Affine3f cam_pose = viz::makeCameraPose(campos.at(pose)*obj_dist+cam_focal_point, cam_focal_point, cam_y_dir*obj_dist+cam_focal_point);
            /* Get the transformation matrix from camera coordinate system to global. */
            Affine3f transform = viz::makeTransformToGlobal(Vec3f(1.0f,0.0f,0.0f), Vec3f(0.0f,1.0f,0.0f), Vec3f(0.0f,0.0f,1.0f), campos.at(pose));
            viz::WMesh mesh_widget(objmesh);
            /* Pose of the widget in camera frame. */
            Affine3f cloud_pose = Affine3f().translate(Vec3f(1.0f,1.0f,1.0f));
            /* Pose of the widget in global frame. */
            Affine3f cloud_pose_global = transform * cloud_pose;

            
            


            /* Save screen shot as images. */
            //myWindow.saveScreenshot(filename);
            cv::Mat img = myWindow.getScreenshot(), inv_img;
            cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
            if( !rgb_use ) 
              cv::cvtColor(img, img, cv::COLOR_RGB2GRAY);
            cv::imwrite(filename, img);
            
            if (binary_out)
            {
            /* Write images into binary files for further using in CNN training. */
                ViewSphere.writeBinaryfile(filename, binaryPath, headerPath,static_cast<int>(campos.size())*num_class, label_class, static_cast<int>(campos.at(pose).x*100), static_cast<int>(campos.at(pose).y*100), static_cast<int>(campos.at(pose).z*100), rgb_use);
            }
            cnt_img++;
        }
    } while (cnt_img != campos.size());
    imglabel.close();
    return 1;
}
