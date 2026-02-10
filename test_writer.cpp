#include <iostream>
#include <fstream>

int main() {
    std::ofstream log_file("rl_training_log.csv");  // Open file
    log_file << "episode,matrix_size,tile,time_ms,reward,q_value\n";
    log_file << "1,512,8,1.25,790.0,395.0\n";
    log_file << "2,512,8,1.22,805.0,600.0\n";
    log_file.close();

    std::cout << "Dummy log file created!\n";
    return 0;
}
