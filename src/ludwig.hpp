#ifndef LUDWIG_HPP__
#define LUDWIG_HPP__

class conjugateGradientSolver
{
  using matrixVectorProduct = decltype(
      GLDSEL::make_program_from_paths(
          boost::hana::make_tuple("", "")
      )
  );

  using dotProduct = decltype(
     GLDSEL::make_program_from_paths(
          boost::hana::make_tuple("", "")
      )
  );

  using saxpy = decltype(
     GLDSEL::make_program_from_paths(
          boost::hana::make_tuple("", ""),
          glDselUniform("factor", float)
      )
  );

  using ratio = decltype(
     GLDSEL::make_program_from_paths(
          boost::hana::make_tuple("", "")
      )
  );

  using copy  = decltype(
     GLDSEL::make_program_from_paths(
          boost::hana::make_tuple("", "")
      )
  );

  using clear  = decltype(
     GLDSEL::make_program_from_paths(
          boost::hana::make_tuple("", "")
      )
  );

  using multAdd = decltype(
     GLDSEL::make_program_from_paths(
          boost::hana::make_tuple("", ""),
          glDselUniform("factor", float)
     )
  );



  GLuint solution_;
  GLuint locks_;
  GLuint residual_;
  GLuint conj_dir_;
  std::array<GLuint, 5> copy_; // intermediate variables for calculations
  size_t iterations_;
  matrixVectorProduct mvprod_;
  dotProduct dotprod_;
  saxpy saxpy_;
  ratio ratio_;
  copy vcopy_;
  multAdd madd_;
  clear clear_;


public:

  conjugateGradientSolver(int);

  ~conjugateGradientSolver();

  conjugateGradientSolver(conjugateGradientSolver &&other);

  conjugateGradientSolver(const conjugateGradientSolver &other) = delete;

  conjugateGradientSolver(conjugateGradientSolver &other) = delete; // no copying, this owns a resource

  void test(int, GLuint, GLuint, GLuint, GLuint);

  void operator()(int dim, GLuint matrix, GLuint output, GLuint x0);

};

#endif
