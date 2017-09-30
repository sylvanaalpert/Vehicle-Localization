/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <limits>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) 
{
    particles.resize(200);
    default_random_engine gen;
    
    // Create normal (Gaussian) distributions 
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_t(theta, std[2]);

    for (auto & p : particles)
    {
        p.x      = dist_x(gen);
        p.y      = dist_y(gen);
        p.theta  = dist_t(gen);
        p.weight = 1.0;
    }

    is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) 
{

    default_random_engine gen;

    for (auto & p : particles)
    {
        // Create normal (Gaussian) distributions 
        normal_distribution<double> dist_x(p.x, std_pos[0]);
        normal_distribution<double> dist_y(p.y, std_pos[1]);
        normal_distribution<double> dist_t(p.theta, std_pos[2]);

        if (abs(yaw_rate) > 0.0001)
        {
            p.x      = dist_x(gen) + ( sin(p.theta + yaw_rate * delta_t) - sin(p.theta) ) * velocity / yaw_rate;
            p.y      = dist_y(gen) + ( cos(p.theta) - cos(p.theta + yaw_rate * delta_t) ) * velocity / yaw_rate;
            p.theta  = dist_t(gen) + yaw_rate * delta_t;
        }
        else
        {
            p.x      = dist_x(gen) + velocity * cos(p.theta) * delta_t;
            p.y      = dist_y(gen) + velocity * sin(p.theta) * delta_t;
            p.theta  = dist_t(gen);
        }
    }

}

void ParticleFilter::dataAssociation(const std::vector<LandmarkObs> & inPredicted, std::vector<LandmarkObs> & ioObservations) 
{

    for (auto & o : ioObservations)
    {
        double min_distance = numeric_limits<double>::max();

        for (auto & p : inPredicted)
        {
            double d = sqrt(pow(o.x - p.x, 2) + pow(o.y - p.y, 2));
            if (d < min_distance)
            {
                o.id = p.id;
                min_distance = d;
            }
        }
    }


}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
        const std::vector<LandmarkObs> & observations, const Map & map_landmarks) 
{

    for (auto & p : particles)
    {
        double weight = 1.0;

        std::vector<LandmarkObs> map_observations = observations;

        // Transform observations to map coords
        for (auto & o : map_observations)
        {
            double xm, ym;
            xm = p.x + cos(p.theta) * o.x - sin(p.theta) * o.y;
            ym = p.y + sin(p.theta) * o.x + cos(p.theta) * o.y;
            o.x = xm;
            o.y = ym;
        }

        // Match observations to map landmarks
        std::vector<LandmarkObs> predictions;
        for (auto & l : map_landmarks.landmark_list)
        {

            double d = sqrt(pow(double(l.x_f) - p.x, 2) + pow(double(l.y_f) - p.y, 2));

            if (d < sensor_range)
            {
                LandmarkObs lo;
                lo.id = l.id_i;
                lo.x = double(l.x_f);
                lo.y = double(l.y_f);

                predictions.push_back(lo);
            }
        }

        dataAssociation(predictions, map_observations);

        // Calculate the probability for each observation and update particle's final observation
        double sig_x = std_landmark[0];
        double sig_y = std_landmark[1];
        double gauss_norm = (1.0/(2 * M_PI * sig_x * sig_y));

        for (auto & o : map_observations)
        {
            double p = 1.0;

            auto result = find_if(predictions.begin(),
                predictions.end(), 
                [&o](const LandmarkObs & l ) { return (o.id == l.id); });

            double mu_x = (*result).x;
            double mu_y = (*result).y;

            double exponent = pow(o.x - mu_x, 2)/(2 * pow(sig_x, 2)) + pow(o.y - mu_y, 2)/(2 * pow(sig_y, 2));
            p = gauss_norm * exp(-exponent);

            weight *= p;
        }

        p.weight = weight;
    }

}

void ParticleFilter::resample() 
{

    std::vector<double> weights(particles.size());

    for (size_t i(0); i < weights.size(); ++i)
    {
        weights[i] = particles[i].weight;
    }

    random_device rd;
    mt19937 gen(rd());
    discrete_distribution<> d(weights.begin(), weights.end());

    std::vector<Particle> new_particles = particles;

    for (size_t i(0); i < particles.size(); ++i)
    {
        new_particles[i] = particles[d(gen)];
    }

    particles = new_particles;

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{

    //Clear the previous associations
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
    vector<int> v = best.associations;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
    vector<double> v = best.sense_x;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
    vector<double> v = best.sense_y;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
