#ifndef DISTVIZ_CPP_FILE_READER_H_
#define DISTVIZ_CPP_FILE_READER_H_

#include "Eigen/Dense"
#include <string>

using namespace std;

namespace hipgraph::distviz::io {
class FileReader {

public:
 static Eigen::MatrixXf load_data(string file_path, int no_of_images,
                                   int dimension, int rank, int world_size);

static vector<vector<VALUE_TYPE>> load_data_into_2D_vector(string file_path, int no_of_images,
                                             int dimension, int rank, int world_size);


 static int reverse_int (int i);

};
}

#endif // DISTVIZ_CPP_FILE_READER_H_
