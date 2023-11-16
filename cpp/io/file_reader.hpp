#ifndef DISTVIZ_CPP_FILE_READER_H_
#define DISTVIZ_CPP_FILE_READER_H_

#include "Eigen/Dense"
#include <string>

using namespace std;

namespace hipgraph::distviz {
class FileReader {

public:
 static Eigen::MatrixXf load_data(string file_path,int dimension);
};
}

#endif // DISTVIZ_CPP_FILE_READER_H_
