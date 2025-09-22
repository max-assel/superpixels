#pragma once

#include <cmath>
#include <cstddef> // For std::size_t

class SuperpixelParams
{
    public:

        // These are all overridden from yaml

        // Floor image parameters
        int k_c_ = 0;                       // Floor width (pixels)
        double v_fov_ = 0.0;                // Vertical field of view (radians)
        double v_offset_ = 0.0;             // Vertical offset of egocan
        double h_ = 0.0;                    // Vertical angle from camera to floor (radians)

        // Dilation parameters
        int num_dilation_iterations_ = 0;   // Number of dilation iterations
        int kernel_radius_ = 0;             // Kernel size for dilation

        // Superpixel algorithm parameters
        int num_iterations_ = 0;            // Number of iterations
        int num_superpixels_ = 0;           // Desired number of approximately equally-sized superpixels
        int step_ = 0;                      // superpixel grid interval
        bool warm_start_ = false;           // Warm start
        bool constraint_ = false;           // Use constraint
        bool ransac_ = false;               // Refine normals via RANSAC
        bool snapping_ = false;             // Snap clusters to nearest actual pixel

        // Superpixel distance parameters
        double w_normal_ = 0.0;             // Weighting parameter for normal similarity term
        double w_pos_ = 0.0;                // Weighting parameter for plane - position distance term
        double w_compact_ = 0.0;            // Weighting parameter for compactness term

        // RANSAC parameters
        size_t ransac_K = 0;                // Number of points to sample
        int ransac_N = 0;                   // Number of iterations
        double ransac_T = 0.0;              // Threshold    
};