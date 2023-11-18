#ifndef DISTVIZ_CPP_CORE_COMMON_H_
#define DISTVIZ_CPP_CORE_COMMON_H_

namespace hipgraph::distviz::knng {

struct LeafPriority {
  int leaf_index;
  float priority;
};

template <typename INDEX_TYPE, typename VALUE_TYPE>
struct DataNode {
  INDEX_TYPE index;
  VALUE_TYPE value;
  vector<VALUE_TYPE> image_data;
};

template <typename INDEX_TYPE, typename VALUE_TYPE>
struct  EdgeNode {
  INDEX_TYPE src_index;
  INDEX_TYPE dst_index;
  VALUE_TYPE distance;
};

template<typename T>
vector <T> slice (vector<T> const &v, int m, int n) {
  auto first = v.cbegin () + m;
  auto last = v.cbegin () + n + 1;
  std::vector <T> vec (first, last);
  return vec;
}

enum class StorageFormat { RAW, COLUMN };

template<class T, class X>
void sortByFreq (std::vector<T> &v, std::vector<X> &vec, int world_size)
{
  std::unordered_map <T, size_t> count;

  for (int i=0; i< v.size();i++)
  {
    count[v[i]]++;
  }

  std::sort (v.begin (), v.end (), [&count] (T const &a, T const &b)
            {
              if (a == b)
              {
                return false;
              }
              if (count[a] > count[b])
              {
                return true;
              }
              else if (count[a] < count[b])
              {
                return false;
              }
              return a < b;
            });
  auto last = std::unique (v.begin (), v.end ());
  v.erase (last, v.end ());

  for (int i=0; i< v.size();i++)
  {
    float priority = static_cast<float>(count[v[i]] / world_size);

    hipgraph::distviz::knng::LeafPriority leaf_priority;
    leaf_priority.priority = priority;
    leaf_priority.leaf_index = v[i];
    vec[v[i]] = leaf_priority;
  }
}



}



#endif // DISTVIZ_CPP_CORE_COMMON_H_
