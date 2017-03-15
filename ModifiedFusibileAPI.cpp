#ifdef _WIN32
#include <windows.h>
#include <ctime>
#include <direct.h>
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <sys/types.h>
#include <dirent.h>


// Includes CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_texture_types.h>
#include <vector_types.h>

#ifdef _MSC_VER
#include <io.h>
#define R_OK 04
#else
#include <unistd.h>
#endif

// CUDA helper functions
#include "helper_cuda.h"         // helper functions for CUDA error check

#include <map> // multimap

#include <sys/stat.h> // mkdir
#include <sys/types.h> // mkdir

#include "algorithmparameters.h"
#include "globalstate.h"
#include "fusibile.h"

#include "main.h"
#include "fileIoUtils.h"
#include "cameraGeometryUtils.h"
#include "mathUtils.h"
#include "displayUtils.h"
#include "point_cloud_list.h"

#define MAX_NR_POINTS 500000

struct InputData
{
  string path;
  //int id;
  string id;
  int camId;
  Camera cam;
  Mat_<float> depthMap;
  Mat_<Vec3b> inputImage;
  Mat_<Vec3f> normals;
};

int getCameraFromId(string id, vector<Camera> &cameras)
{
  for (size_t i = 0; i < cameras.size(); i++)
  {
    //cout << "Checking camera id " << i << " cameraid " << cameras[i].id << endl;
    if (cameras[i].id.compare(id) == 0)
    {
      return i;
    }
  }
  return -1;
}
static void get_subfolders(
    const char *dirname,
    vector<string> &subfolders)
{
  DIR *dir;
  struct dirent *ent;

  // Open directory stream
  dir = opendir(dirname);
  if (dir != NULL)
  {
    //cout << "Dirname is " << dirname << endl;
    //cout << "Dirname type is " << ent->d_type << endl;
    //cout << "Dirname type DT_DIR " << DT_DIR << endl;

    // Print all files and directories within the directory
    while ((ent = readdir(dir)) != NULL)
    {
      //cout << "INSIDE" << endl;
      //if(ent->d_type == DT_DIR)
      {
        char *name = ent->d_name;
        if (strcmp(name, ".") == 0 || strcmp(ent->d_name, "..") == 0)
        {
          continue;
        }
        //printf ("dir %s/\n", name);
        subfolders.push_back(string(name));
      }
    }

    closedir(dir);

  }
  else
  {
    // Could not open directory
    printf("Cannot open directory %s\n", dirname);
    exit(EXIT_FAILURE);
  }
}

static void print_help()
{
  printf("\nfusibile\n");
}

/* process command line arguments
 * Input: argc, argv - command line arguments
 * Output: inputFiles, outputFiles, parameters, gt_parameters, no_display - algorithm parameters
 */
static int getParametersFromCommandLine(int argc,
                                        char **argv,
                                        InputFiles &inputFiles,
                                        OutputFiles &outputFiles,
                                        AlgorithmParameters &parameters,
                                        GTcheckParameters &gt_parameters,
                                        bool &no_display)
{
  const char *algorithm_opt = "--algorithm=";
  const char *maxdisp_opt = "--max-disparity=";
  const char *blocksize_opt = "--blocksize=";
  const char *cost_tau_color_opt = "--cost_tau_color=";
  const char *cost_tau_gradient_opt = "--cost_tau_gradient=";
  const char *cost_alpha_opt = "--cost_alpha=";
  const char *cost_gamma_opt = "--cost_gamma=";
  const char *disparity_tolerance_opt = "--disp_tol=";
  const char *normal_tolerance_opt = "--norm_tol=";
  const char *border_value = "--border_value="; //either constant scalar or -1 = REPLICATE
  const char *gtDepth_divFactor_opt = "--gtDepth_divisionFactor=";
  const char *gtDepth_tolerance_opt = "--gtDepth_tolerance=";
  const char *gtDepth_tolerance2_opt = "--gtDepth_tolerance2=";
  const char *nodisplay_opt = "-no_display";
  const char *colorProc_opt = "-color_processing";
  const char *num_iterations_opt = "--iterations=";
  const char *self_similariy_n_opt = "--ss_n=";
  const char *ct_epsilon_opt = "--ct_eps=";
  const char *cam_scale_opt = "--cam_scale=";
  const char *num_img_processed_opt = "--num_img_processed=";
  const char *n_best_opt = "--n_best=";
  const char *cost_comb_opt = "--cost_comb=";
  const char *cost_good_factor_opt = "--good_factor=";
  const char *depth_min_opt = "--depth_min=";
  const char *depth_max_opt = "--depth_max=";
  //    const char* scale_opt         = "--scale=";
  const char *outputPath_opt = "-output_folder";
  const char *calib_opt = "-calib_file";
  const char *gt_opt = "-gt";
  const char *gt_nocc_opt = "-gt_nocc";
  const char *occl_mask_opt = "-occl_mask";
  const char *gt_normal_opt = "-gt_normal";
  const char *images_input_folder_opt = "-images_folder";
  const char *p_input_folder_opt = "-p_folder";
  const char *krt_file_opt = "-krt_file";
  const char *camera_input_folder_opt = "-camera_folder";
  const char *bounding_folder_opt = "-bounding_folder";
  const char *viewSelection_opt = "-view_selection";
  const char *initial_seed_opt = "--initial_seed";

  const char *disp_thresh_opt = "--disp_thresh=";
  const char *normal_thresh_opt = "--normal_thresh=";
  const char *num_consistent_opt = "--num_consistent=";

  //read in arguments
  for (int i = 1; i < argc; i++)
  {
    if (argv[i][0] != '-')
    {
      inputFiles.img_filenames.push_back(argv[i]);
    }
    else if (strncmp(argv[i], algorithm_opt, strlen(algorithm_opt)) == 0)
    {
      char *_alg = argv[i] + strlen(algorithm_opt);
      parameters.algorithm = strcmp(_alg, "pm") == 0 ? PM_COST :
                             strcmp(_alg, "ct") == 0 ? CENSUS_TRANSFORM :
                             strcmp(_alg, "sct") == 0 ? SPARSE_CENSUS :
                             strcmp(_alg, "ct_ss") == 0 ? CENSUS_SELFSIMILARITY :
                             strcmp(_alg, "adct") == 0 ? ADCENSUS :
                             strcmp(_alg, "adct_ss") == 0 ? ADCENSUS_SELFSIMILARITY :
                             strcmp(_alg, "pm_ss") == 0 ? PM_SELFSIMILARITY : -1;
      if (parameters.algorithm < 0)
      {
        printf("Command-line parameter error: Unknown stereo algorithm\n\n");
        print_help();
        return -1;
      }
    }
    else if (strncmp(argv[i], cost_comb_opt, strlen(cost_comb_opt)) == 0)
    {
      char *_alg = argv[i] + strlen(algorithm_opt);
      parameters.cost_comb = strcmp(_alg, "all") == 0 ? COMB_ALL :
                             strcmp(_alg, "best_n") == 0 ? COMB_BEST_N :
                             strcmp(_alg, "angle") == 0 ? COMB_ANGLE :
                             strcmp(_alg, "good") == 0 ? COMB_GOOD : -1;
      if (parameters.cost_comb < 0)
      {
        printf("Command-line parameter error: Unknown cost combination method\n\n");
        print_help();
        return -1;
      }
    }
    else if (strncmp(argv[i], maxdisp_opt, strlen(maxdisp_opt)) == 0)
    {
      if (sscanf(argv[i] + strlen(maxdisp_opt), "%f", &parameters.max_disparity) != 1 ||
          parameters.max_disparity < 1)
      {
        printf("Command-line parameter error: The max disparity (--maxdisparity=<...>) must be a positive integer \n");
        print_help();
        return -1;
      }
    }
    else if (strncmp(argv[i], blocksize_opt, strlen(blocksize_opt)) == 0)
    {
      int k_size;
      if (sscanf(argv[i] + strlen(blocksize_opt), "%d", &k_size) != 1 ||
          k_size < 1 || k_size % 2 != 1)
      {
        printf("Command-line parameter error: The block size (--blocksize=<...>) must be a positive odd number\n");
        return -1;
      }
      parameters.box_hsize = k_size;
      parameters.box_vsize = k_size;
    }
    else if (strncmp(argv[i], cost_good_factor_opt, strlen(cost_good_factor_opt)) == 0)
    {
      sscanf(argv[i] + strlen(cost_good_factor_opt), "%f", &parameters.good_factor);
    }
    else if (strncmp(argv[i], cost_tau_color_opt, strlen(cost_tau_color_opt)) == 0)
    {
      sscanf(argv[i] + strlen(cost_tau_color_opt), "%f", &parameters.tau_color);
    }
    else if (strncmp(argv[i], cost_tau_gradient_opt, strlen(cost_tau_gradient_opt)) == 0)
    {
      sscanf(argv[i] + strlen(cost_tau_gradient_opt), "%f", &parameters.tau_gradient);
    }
    else if (strncmp(argv[i], cost_alpha_opt, strlen(cost_alpha_opt)) == 0)
    {
      sscanf(argv[i] + strlen(cost_alpha_opt), "%f", &parameters.alpha);
    }
    else if (strncmp(argv[i], cost_gamma_opt, strlen(cost_gamma_opt)) == 0)
    {
      sscanf(argv[i] + strlen(cost_gamma_opt), "%f", &parameters.gamma);
    }
    else if (strncmp(argv[i], border_value, strlen(border_value)) == 0)
    {
      sscanf(argv[i] + strlen(border_value), "%d", &parameters.border_value);
    }
    else if (strncmp(argv[i], num_iterations_opt, strlen(num_iterations_opt)) == 0)
    {
      sscanf(argv[i] + strlen(num_iterations_opt), "%d", &parameters.iterations);
    }
    else if (strncmp(argv[i], disparity_tolerance_opt, strlen(disparity_tolerance_opt)) == 0)
    {
      sscanf(argv[i] + strlen(disparity_tolerance_opt), "%f", &parameters.dispTol);
    }
    else if (strncmp(argv[i], normal_tolerance_opt, strlen(normal_tolerance_opt)) == 0)
    {
      sscanf(argv[i] + strlen(normal_tolerance_opt), "%f", &parameters.normTol);
    }
    else if (strncmp(argv[i], self_similariy_n_opt, strlen(self_similariy_n_opt)) == 0)
    {
      sscanf(argv[i] + strlen(self_similariy_n_opt), "%d", &parameters.self_similarity_n);
    }
    else if (strncmp(argv[i], ct_epsilon_opt, strlen(ct_epsilon_opt)) == 0)
    {
      sscanf(argv[i] + strlen(ct_epsilon_opt), "%f", &parameters.census_epsilon);
    }
    else if (strncmp(argv[i], cam_scale_opt, strlen(cam_scale_opt)) == 0)
    {
      sscanf(argv[i] + strlen(cam_scale_opt), "%f", &parameters.cam_scale);
    }
    else if (strncmp(argv[i], num_img_processed_opt, strlen(num_img_processed_opt)) == 0)
    {
      sscanf(argv[i] + strlen(num_img_processed_opt), "%d", &parameters.num_img_processed);
    }
    else if (strncmp(argv[i], n_best_opt, strlen(n_best_opt)) == 0)
    {
      sscanf(argv[i] + strlen(n_best_opt), "%d", &parameters.n_best);
    }
    else if (strncmp(argv[i], gtDepth_divFactor_opt, strlen(gtDepth_divFactor_opt)) == 0)
    {
      sscanf(argv[i] + strlen(gtDepth_divFactor_opt), "%f", &gt_parameters.divFactor);
    }
    else if (strncmp(argv[i], gtDepth_tolerance_opt, strlen(gtDepth_tolerance_opt)) == 0)
    {
      sscanf(argv[i] + strlen(gtDepth_tolerance_opt), "%f", &gt_parameters.dispTolGT);
    }
    else if (strncmp(argv[i], gtDepth_tolerance2_opt, strlen(gtDepth_tolerance2_opt)) == 0)
    {
      sscanf(argv[i] + strlen(gtDepth_tolerance2_opt), "%f", &gt_parameters.dispTolGT2);
    }
    else if (strncmp(argv[i], depth_min_opt, strlen(depth_min_opt)) == 0)
    {
      sscanf(argv[i] + strlen(depth_min_opt), "%f", &parameters.depthMin);
    }
    else if (strncmp(argv[i], depth_max_opt, strlen(depth_max_opt)) == 0)
    {
      sscanf(argv[i] + strlen(depth_max_opt), "%f", &parameters.depthMax);
    }
    else if (strcmp(argv[i], viewSelection_opt) == 0)
    {
      parameters.viewSelection = true;
    }
    else if (strcmp(argv[i], nodisplay_opt) == 0)
    {
      no_display = true;
    }
    else if (strcmp(argv[i], colorProc_opt) == 0)
    {
      parameters.color_processing = true;
    }
    else if (strcmp(argv[i], "-o") == 0)
    {
      outputFiles.disparity_filename = argv[++i];
    }
    else if (strcmp(argv[i], outputPath_opt) == 0)
    {
      outputFiles.parentFolder = argv[++i];
    }
    else if (strcmp(argv[i], calib_opt) == 0)
    {
      inputFiles.calib_filename = argv[++i];
    }
    else if (strcmp(argv[i], gt_opt) == 0)
    {
      inputFiles.gt_filename = argv[++i];
    }
    else if (strcmp(argv[i], gt_nocc_opt) == 0)
    {
      inputFiles.gt_nocc_filename = argv[++i];
    }
    else if (strcmp(argv[i], occl_mask_opt) == 0)
    {
      inputFiles.occ_filename = argv[++i];
    }
    else if (strcmp(argv[i], gt_normal_opt) == 0)
    {
      inputFiles.gt_normal_filename = argv[++i];
    }
    else if (strcmp(argv[i], images_input_folder_opt) == 0)
    {
      inputFiles.images_folder = argv[++i];
    }
    else if (strcmp(argv[i], p_input_folder_opt) == 0)
    {
      inputFiles.p_folder = argv[++i];
    }
    else if (strcmp(argv[i], krt_file_opt) == 0)
    {
      inputFiles.krt_file = argv[++i];
    }
    else if (strcmp(argv[i], camera_input_folder_opt) == 0)
    {
      inputFiles.camera_folder = argv[++i];
    }
    else if (strcmp(argv[i], initial_seed_opt) == 0)
    {
      inputFiles.seed_file = argv[++i];
    }
    else if (strcmp(argv[i], bounding_folder_opt) == 0)
    {
      inputFiles.bounding_folder = argv[++i];
    }
    else if (strncmp(argv[i], disp_thresh_opt, strlen(disp_thresh_opt)) == 0)
    {
      sscanf(argv[i] + strlen(disp_thresh_opt), "%f", &parameters.depthThresh);
    }
    else if (strncmp(argv[i], normal_thresh_opt, strlen(normal_thresh_opt)) == 0)
    {
      float angle_degree;
      sscanf(argv[i] + strlen(normal_thresh_opt), "%f", &angle_degree);
      parameters.normalThresh = angle_degree * M_PI / 180.0f;
    }
    else if (strncmp(argv[i], num_consistent_opt, strlen(num_consistent_opt)) == 0)
    {
      sscanf(argv[i] + strlen(num_consistent_opt), "%d", &parameters.numConsistentThresh);
    }
    else
    {
      printf("Command-line parameter error: unknown option %s\n", argv[i]);
    }
  }
  return 0;
}

static void addImageToTextureFloatColor(vector<Mat> &imgs, cudaTextureObject_t texs[])
{
  for (size_t i = 0; i < imgs.size(); i++)
  {
    int rows = imgs[i].rows;
    int cols = imgs[i].cols;
    // Create channel with floating point type
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();

    // Allocate array with correct size and number of channels
    cudaArray *cuArray;
    checkCudaErrors(cudaMallocArray(&cuArray,
                                    &channelDesc,
                                    cols,
                                    rows));

    checkCudaErrors (cudaMemcpy2DToArray(cuArray,
                                         0,
                                         0,
                                         imgs[i].ptr<float>(),
                                         imgs[i].step[0],
                                         cols * sizeof(float) * 4,
                                         rows,
                                         cudaMemcpyHostToDevice));

    // Specify texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    // Specify texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    // Create texture object
    checkCudaErrors(cudaCreateTextureObject(&(texs[i]), &resDesc, &texDesc, NULL));
  }
  return;
}

static int modifiedRunFusibile(std::vector<cv::Mat> colorImages,
                               std::vector<cv::Mat> normalImages,
                               std::vector<cv::Mat> depthImages,
                               std::vector<cv::Matx34f> pMatrices,
                               AlgorithmParameters &algParameters,
                               std::vector<std::pair<cv::Vec4f, cv::Vec3f> > &pointsResults)
{

  string ext = ".png";

  algParameters.remove_black_background = true;

  GTcheckParameters gtParameters;

  gtParameters.dispTolGT = 0.1f;
  gtParameters.dispTolGT2 = 0.02f;
  gtParameters.divFactor = 1.0f;
  // create folder to store result images
  time_t timeObj;
  time(&timeObj);

  vector<Mat_<Vec3f> > view_vectors;

  time(&timeObj);

  size_t numImages = colorImages.size();
  cout << "numImages is " << numImages << endl;

  algParameters.num_img_processed = min((int) numImages, algParameters.num_img_processed);

  vector<Mat_<Vec3f> > img_color;

  for (size_t i = 0; i < numImages; i++)
  {
    img_color.push_back(cv::Mat_<Vec3f>(colorImages.at(i)));
//    img_color.push_back(imread((inputFiles.images_folder + inputFiles.img_filenames[i]), IMREAD_COLOR));
  }

  // Actual fusibile call
  size_t avail;
  size_t total;
  cudaMemGetInfo(&avail, &total);
  size_t used = total - avail;
  printf("Device memory used: %fMB\n", used / 1000000.0f);

  GlobalState *gs = new GlobalState;
  gs->cameras = new CameraParameters_cu;
  gs->pc = new PointCloud;
  cudaMemGetInfo(&avail, &total);
  used = total - avail;
  printf("Device memory used: %fMB\n", used / 1000000.0f);

  uint32_t rows = img_color[0].rows;
  uint32_t cols = img_color[0].cols;

  CameraParameters camParams = getCameraParameters(*(gs->cameras),
                                                   pMatrices, algParameters.depthMin,
                                                   algParameters.depthMax,
                                                   algParameters.cam_scale,
                                                   false);
  printf("Camera size is %lu\n", camParams.cameras.size());

  for (int i = 0; i < algParameters.num_img_processed; i++)
  {
    algParameters.min_disparity =
        disparityDepthConversion(camParams.f, camParams.cameras[i].baseline, camParams.cameras[i].depthMax);
    algParameters.max_disparity =
        disparityDepthConversion(camParams.f, camParams.cameras[i].baseline, camParams.cameras[i].depthMin);
  }

  int numSelViews = colorImages.size();
  gs->cameras->viewSelectionSubsetNumber = numSelViews;

  for (int i = 0; i < numSelViews; i++)
  {
    gs->cameras->viewSelectionSubset[i] = camParams.viewSelectionSubset[i];
  }

  // run gpu run
  // Init parameters
  gs->params = &algParameters;

  // Init ImageInfo
  gs->cameras->cols = img_color[0].cols;
  gs->cameras->rows = img_color[0].rows;
  gs->params->cols = img_color[0].cols;
  gs->params->rows = img_color[0].rows;
  gs->resize(img_color.size());
  gs->pc->resize(img_color[0].rows * img_color[0].cols);
  PointCloudList pc_list;
  pc_list.resize(img_color[0].rows * img_color[0].cols);
  pc_list.size = 0;
  pc_list.rows = img_color[0].rows;
  pc_list.cols = img_color[0].cols;
  gs->pc->rows = img_color[0].rows;
  gs->pc->cols = img_color[0].cols;

  // Resize lines
  for (size_t i = 0; i < img_color.size(); i++)
  {
    gs->lines[i].resize(img_color[0].rows * img_color[0].cols);
    gs->lines[i].n = img_color[0].rows * img_color[0].cols;
    gs->lines[i].s = img_color[0].cols;
    gs->lines[i].l = img_color[0].cols;
  }

  vector<Mat> img_color_float(img_color.size());
  vector<Mat> img_color_float_alpha(img_color.size());
  vector<Mat> normals_and_depth(img_color.size());

  for (size_t i = 0; i < img_color.size(); i++)
  {
    vector<Mat_<float> > rgbChannels(3);
    img_color_float_alpha[i] = Mat::zeros(img_color[0].rows, img_color[0].cols, CV_32FC4);
    img_color[i].convertTo(img_color_float[i], CV_32FC3); // or CV_32F works (too)
    Mat alpha(img_color[0].rows, img_color[0].cols, CV_32FC1);
    split(img_color_float[i], rgbChannels);
    rgbChannels.push_back(alpha);
    merge(rgbChannels, img_color_float_alpha[i]);

    /* Create vector of normals and disparities */
    vector<Mat_<float> > normal(3);
    normals_and_depth[i] = Mat::zeros(img_color[0].rows, img_color[0].cols, CV_32FC4);
    split(normalImages[i], normal);
    normal.push_back(depthImages[i]);
    merge(normal, normals_and_depth[i]);

  }

  addImageToTextureFloatColor(img_color_float_alpha, gs->imgs);

  addImageToTextureFloatColor(normals_and_depth, gs->normals_depths);

#define pow2(x) ((x)*(x))
#define get_pow2_norm(x, y) (pow2(x)+pow2(y))

  runcuda(*gs, pc_list, numSelViews);

  pointsResults = std::vector<std::pair<cv::Vec4f, cv::Vec3f> >(pc_list.size);
  for (unsigned int index = 0; index < pc_list.size; ++index)
  {
    pointsResults.push_back(
        std::make_pair(
            cv::Vec4f(pc_list.points[index].coord.x,
                      pc_list.points[index].coord.y,
                      pc_list.points[index].coord.z,
                      pc_list.points[index].coord.w),
            cv::Vec3f(pc_list.points[index].texture.x,
                      pc_list.points[index].texture.y,
                      pc_list.points[index].texture.z)));
  }

  return 0;
}

int main(int argc, char **argv)
{
  if (argc < 3)
  {
    print_help();
    return 0;
  }

  InputFiles inputFiles;
  OutputFiles outputFiles;
  AlgorithmParameters *algParameters = new AlgorithmParameters;
  GTcheckParameters gtParameters;
  bool no_display = false;

  int ret = getParametersFromCommandLine(argc, argv, inputFiles, outputFiles, *algParameters, gtParameters, no_display);
  if (ret != 0)
  {
    return ret;
  }

  std::vector<std::pair<cv::Vec4f,cv::Vec3f> > results;
  std::vector<cv::Mat> colorImages;
  std::vector<cv::Mat> normalImages;
  std::vector<cv::Mat> depthImages;
  std::vector<cv::Matx34f> pMatrices;

  modifiedRunFusibile(colorImages,normalImages,depthImages,pMatrices,*algParameters, results);

  return 0;
}

