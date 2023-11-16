#ifndef DISTVIZ_CPP_FILE_READER_H_
#define DISTVIZ_CPP_FILE_READER_H_

#include "Eigen/Dense"
#include <string>

namespace hipgraph::distviz {
class FileReader {

public:
  Eigen::MatrixXf load_data(string file_path);
};
}

#endif // DISTVIZ_CPP_FILE_READER_H_
