#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "program.hpp"
#include "ludwig.hpp"

conjugateGradientSolver::conjugateGradientSolver(int iterations) :
  solution_(0),
  locks_(0),
  residual_(0),
  conj_dir_(0),
  copy_(),
  iterations_(iterations),
  mvprod_(GLDSEL::make_program_from_paths(
                                          boost::hana::make_tuple(boost::none,boost::none,boost::none,boost::none,boost::none, "../src/mvprod/main.cs")
                                          )
          ),
  dotprod_(GLDSEL::make_program_from_paths(
                                           boost::hana::make_tuple(boost::none,boost::none,boost::none,boost::none,boost::none, "../src/dotprod/main.cs")
                                           )
           ),
  saxpy_(GLDSEL::make_program_from_paths(
                                         boost::hana::make_tuple(boost::none,boost::none,boost::none,boost::none,boost::none, "../src/initial/main.cs"),
                                           glDselUniform("factor", float)
                                           )
         ),
  ratio_(GLDSEL::make_program_from_paths(
                                           boost::hana::make_tuple(boost::none,boost::none,boost::none,boost::none,boost::none, "../src/ratio/main.cs")
                                           )
         ),
  vcopy_(GLDSEL::make_program_from_paths(
                                           boost::hana::make_tuple(boost::none,boost::none,boost::none,boost::none,boost::none, "../src/copy/main.cs")
                                           )
         ),
  madd_(GLDSEL::make_program_from_paths(
                                         boost::hana::make_tuple(boost::none,boost::none,boost::none,boost::none,boost::none, "../src/multAdd/main.cs"),
                                         glDselUniform("factor", float)
                                         )
        )
{
  glGenTextures(1, &solution_);
  glBindTexture(GL_TEXTURE_1D, solution_);
  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  glGenTextures(1, &locks_);
  glBindTexture(GL_TEXTURE_1D, locks_);
  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  glGenTextures(1, &residual_);
  glBindTexture(GL_TEXTURE_1D, residual_);
  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  glGenTextures(1, &conj_dir_);
  glBindTexture(GL_TEXTURE_1D, conj_dir_);
  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  for(auto i = size_t{0}; i < copy_.size(); i++)
  {
    glGenTextures(1, &copy_[i]);
    glBindTexture(GL_TEXTURE_1D, copy_[i]);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  }
}

conjugateGradientSolver::conjugateGradientSolver(conjugateGradientSolver &&other) :
  solution_(other.solution_),
  locks_(other.locks_),
  residual_(other.residual_),
  conj_dir_(other.conj_dir_),
  copy_(other.copy_),
  iterations_(other.iterations_),
  mvprod_(std::move(other.mvprod_)),
  dotprod_(std::move(other.dotprod_)),
  saxpy_(std::move(other.saxpy_)),
  ratio_(std::move(other.ratio_)),
  vcopy_(std::move(other.vcopy_)),
  madd_(std::move(madd_))
{
  other.solution_ = 0;
  other.locks_ = 0;
  other.copy_[0] = 0;
}

conjugateGradientSolver::~conjugateGradientSolver()
{
  glDeleteTextures(1, &solution_);
  glDeleteTextures(1, &locks_);
  glDeleteTextures(1, &copy_[0]);
}

void conjugateGradientSolver::operator()(int dim, GLuint matrix, GLuint output, GLuint x0)
{
  // resize locks, target, and copy to dim
  std::vector<int> lockdata(dim, 0);

  glBindTexture(GL_TEXTURE_1D, locks_);
  glTexImage1D(GL_TEXTURE_1D, 0, GL_R32I, dim, 0, GL_RED_INTEGER, GL_INT, lockdata.data());
  glBindTexture(GL_TEXTURE_1D, solution_);
  glTexImage1D(GL_TEXTURE_1D, 0, GL_R32F, dim, 0, GL_RED, GL_FLOAT, nullptr);
  glBindTexture(GL_TEXTURE_1D, residual_);
  glTexImage1D(GL_TEXTURE_1D, 0, GL_R32F, dim, 0, GL_RED, GL_FLOAT, nullptr);
  glBindTexture(GL_TEXTURE_1D, conj_dir_);
  glTexImage1D(GL_TEXTURE_1D, 0, GL_R32F, dim, 0, GL_RED, GL_FLOAT, nullptr);
  for(auto i = size_t{0}; i < copy_.size(); i++)
    {
    glBindTexture(GL_TEXTURE_1D, copy_[i]);
    glTexImage1D(GL_TEXTURE_1D, 0, GL_R32F, dim, 0, GL_RED, GL_FLOAT, nullptr);
  }

  glBindImageTexture(3, residual_, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32F);
  glBindImageTexture(0, matrix, 0, GL_TRUE, 0, GL_READ_ONLY, GL_R32F);
  glBindImageTexture(1, x0, 0, GL_TRUE, 0, GL_READ_ONLY, GL_R32F);
  glBindImageTexture(2, output, 0, GL_TRUE, 0, GL_READ_ONLY, GL_R32F);
  glBindImageTexture(4, locks_, 0, GL_TRUE, 0, GL_READ_ONLY, GL_R32I);

  saxpy_.setUniforms(glDselArgument("factor", -1.0f));

  glDispatchCompute(dim, dim, 1); // calculate b - Ar

  glBindImageTexture(0, residual_, 0, GL_TRUE, 0, GL_READ_ONLY, GL_R32F);
  glBindImageTexture(1, conj_dir_, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32F);

  vcopy_.activate();
  glDispatchCompute(dim, 1, 1); // initialize p_0 to r_0

  for(auto i = size_t{0}; i < iterations_; i++)
  {
    // compute r^T r / p^T A p
    glBindImageTexture(0, residual_, 0, GL_TRUE, 0, GL_READ_ONLY, GL_R32F);
    glBindImageTexture(1, residual_, 0, GL_TRUE, 0, GL_READ_ONLY, GL_R32F);
    glBindImageTexture(2, copy_[0], 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32F);

    dotprod_.activate();

    glDispatchCompute(dim, 1, 1); // copy 0 now contains r^T r

    // compute the divisor

    // step 1: compute the matvec product Ap
    glBindImageTexture(0, matrix, 0, GL_TRUE, 0, GL_READ_ONLY, GL_R32F);
    glBindImageTexture(1, conj_dir_, 0, GL_TRUE, 0, GL_READ_ONLY, GL_R32F);
    glBindImageTexture(2, copy_[1], 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32F);
    glBindImageTexture(3, locks_, 0, GL_TRUE, 0, GL_READ_WRITE, GL_R32I);

    mvprod_.activate();

    glDispatchCompute(dim, dim, 1);
    // now copy_[1] has Ap

    // dot product p and Ap
    glBindImageTexture(0, solution_, 0, GL_TRUE, 0, GL_READ_ONLY, GL_R32F);
    glBindImageTexture(1, copy_[1], 0, GL_TRUE, 0, GL_READ_ONLY, GL_R32F);
    glBindImageTexture(2, copy_[2], 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32F);

    dotprod_.activate();

    glDispatchCompute(dim, 1, 1);
    //copy_[2] has p(Ap)

    glBindImageTexture(0, copy_[0], 0, GL_TRUE, 0, GL_READ_ONLY, GL_R32F);
    glBindImageTexture(1, copy_[2], 0, GL_TRUE, 0, GL_READ_ONLY, GL_R32F);
    glBindImageTexture(2, copy_[4], 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32F);

    ratio_.activate();

    glDispatchCompute(1,1,1);

    // alpha in copy_[4]

    // time to calculate the next steps solution_
    // x_k+1 = x_k + a_k * p_k
    glBindImageTexture(0, conj_dir_, 0, GL_TRUE, 0, GL_READ_ONLY, GL_R32F);
    glBindImageTexture(1, solution_, 0, GL_TRUE, 0, GL_READ_ONLY, GL_R32F);
    glBindImageTexture(2, copy_[2], 0, GL_TRUE, 0, GL_READ_ONLY, GL_R32F);
    glBindImageTexture(3, copy_[3], 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32F);

    madd_.setUniforms(glDselArgument("factor", 1.0f));

    glDispatchCompute(dim, 1, 1); // copy_[3] now contains x_k+1
    std::swap(copy_[3], solution_); // solution_ is updated

    // calculate new residual_
    // r_k+1 = r_k - a_k Ap_k
    glBindImageTexture(0, copy_[1], 0, GL_TRUE, 0, GL_READ_ONLY, GL_R32F);
    glBindImageTexture(1, residual_, 0, GL_TRUE, 0, GL_READ_ONLY, GL_R32F);
    glBindImageTexture(2, copy_[3], 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32F);

    madd_.setUniforms(glDselArgument("factor", -1.0f));

    glDispatchCompute(dim, 1, 1);

    std::swap(copy_[3], residual_); // residual_ is r_k+1
    if(i == (iterations_-1)) // last iter, dont calculate new conjugate dir
    {
      continue;
    }

    glBindImageTexture(0, residual_, 0, GL_TRUE, 0, GL_READ_ONLY, GL_R32F);
    glBindImageTexture(1, residual_, 0, GL_TRUE, 0, GL_READ_ONLY, GL_R32F);
    glBindImageTexture(2, copy_[3], 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32F);

    dotprod_.activate();

    glDispatchCompute(dim, 1, 1);

    glBindImageTexture(0, copy_[3], 0, GL_TRUE, 0, GL_READ_ONLY, GL_R32F);
    glBindImageTexture(1, copy_[0], 0, GL_TRUE, 0, GL_READ_ONLY, GL_R32F);
    glBindImageTexture(2, copy_[1], 0, GL_TRUE, 0, GL_READ_ONLY, GL_R32F);

    ratio_.activate();

    glDispatchCompute(1,1,1);
    // copy 1 now contains beta

    // time to calculate next conj dir
    glBindImageTexture(0, conj_dir_, 0, GL_TRUE, 0, GL_READ_ONLY, GL_R32F);
    glBindImageTexture(1, residual_, 0, GL_TRUE, 0, GL_READ_ONLY, GL_R32F);
    glBindImageTexture(2, copy_[1], 0, GL_TRUE, 0, GL_READ_ONLY, GL_R32F);
    glBindImageTexture(3, copy_[3], 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32F);

    madd_.setUniforms(glDselArgument("factor", 1.0f));

    glDispatchCompute(dim, 1, 1); // fin
  }

  glBindImageTexture(0, solution_, 0, GL_TRUE, 0, GL_READ_ONLY, GL_R32F);
  glBindImageTexture(1, x0, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32F);

  vcopy_.activate();
  glDispatchCompute(dim, 1, 1); // output
}
