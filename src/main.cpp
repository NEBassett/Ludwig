#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <stdlib.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <tuple>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <array>
#include <vector>
#include <complex>
#include <math.h>
#include <utility>
#include <algorithm>
#include <boost/hana/tuple.hpp>
#include <boost/hana/zip.hpp>
#include <boost/hana/for_each.hpp>
#include <boost/hana/ext/std/tuple.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <algorithm>
#include "program.hpp"
#include "ludwig.hpp"

void GLAPIENTRY msgCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam )
{
  std::cout << message << '\n';
}

struct screenQuad
{
  GLuint vao, vbo;

  screenQuad()
  {
    static auto verts = std::array<glm::vec4, 6>{
            glm::vec4(1.0f,  1.0f, 1.0f, 1.0f),
            glm::vec4(1.0f, -1.0f, 1.0f, 0.0f),
            glm::vec4(-1.0f, 1.0f, 0.0f, 1.0f),
            glm::vec4(-1.0f, -1.0f, 0.0f, 0.0f),
            glm::vec4(1.0f, -1.0f, 1.0f, 0.0f),
            glm::vec4(-1.0f, 1.0f, 0.0f, 1.0f)
    };

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec4)*6, verts.data(), GL_STATIC_DRAW);

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (void*)0);
    glEnableVertexAttribArray(0);
  }

  screenQuad(const screenQuad &) = delete;

  ~screenQuad()
  {
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
  }
};






int main()
{
  glfwSetErrorCallback([](auto err, const auto* desc){ std::cout << "Error: " << desc << '\n'; });

  // glfw init
  if(!glfwInit())
  {
    std::cout << "glfw failed to initialize\n";
    std::exit(1);
  }

  // context init
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  auto window = glfwCreateWindow(640, 480, "Lattice Boltzmann Methods", NULL, NULL);
  if (!window)
  {
    std::cout << "window/glcontext failed to initialize\n";
    std::exit(1);
  }

  glfwMakeContextCurrent(window);

  // glew init
  auto err = glewInit();
  if(GLEW_OK != err)
  {
    std::cout << "glew failed to init: " << glewGetErrorString(err) << '\n';
    std::exit(1);
  }

  // gl init
  glEnable(GL_DEBUG_OUTPUT);
  glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
  glPointSize(15.5f);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glDebugMessageCallback(msgCallback, 0);
  glEnable(GL_DEPTH_TEST);



  // program initialization


  int width, height;
  double time = glfwGetTime();
  glm::mat4 proj;
  double tdelta;
  double temp;

  //glfwSetWindowUserPointer(window, &fsim);

   glfwSetKeyCallback(window, [](auto *window, auto key, auto, auto action, auto mods){
     //auto fsimptr = static_cast<>(glfwGetWindowUserPointer(window));

     if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
       glfwSetWindowShouldClose(window, GLFW_TRUE);
     if(key == GLFW_KEY_F4)
     { /* later */ }
   });

   screenQuad quad;

   auto draw = GLDSEL::make_program_from_paths(
    boost::hana::make_tuple("../src/main/vertex.vs", "../src/main/fragment.fs"),
    glDselUniform("time", float),
    glDselUniform("model", glm::mat4),
    glDselUniform("view", glm::mat4),
    glDselUniform("proj", glm::mat4)
  );

   constexpr int dim = 3;

   conjugateGradientSolver cgsolver(10);

   GLuint tex[3];

   glGenTextures(3, tex);

   std::vector<float> initial({0,0,0});
   std::vector<float> target({3,4,5});
   std::vector<float> system({2,-1,0,-1,2,-1,0,-1,2});

   glBindTexture(GL_TEXTURE_1D, tex[0]);
   glTexImage1D(GL_TEXTURE_1D, 0, GL_R32F, dim, 0, GL_RED, GL_FLOAT, initial.data());

   glBindTexture(GL_TEXTURE_1D, tex[1]);
   glTexImage1D(GL_TEXTURE_1D, 0, GL_R32F, dim, 0, GL_RED, GL_FLOAT, target.data());

   glBindTexture(GL_TEXTURE_2D, tex[2]);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, dim, dim, 0, GL_RED, GL_FLOAT, system.data());

   cgsolver(dim, tex[2], tex[1], tex[0]);

   glfwSwapInterval(1);
   while(!glfwWindowShouldClose(window))
   {
    auto oldT = time;
    time = glfwGetTime();

    glfwGetFramebufferSize(window, &width, &height);
    glViewport(0, 0, width, height);
    proj = glm::perspective(1.57f, float(width)/float(height), 0.1f, 7000.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    //draw(proj, view, glm::mat4(), glfwGetTime(), plane);
    draw.setUniforms( // set uniforms
      glDselArgument("model", glm::mat4()),
      glDselArgument("view", glm::mat4()),
      glDselArgument("proj", proj),
      glDselArgument("time", float(glfwGetTime()))
    );

    glBindVertexArray(quad.vao);

    glDrawArrays(GL_TRIANGLES, 0, 6); // draw quad

    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  glfwTerminate();
  return 0;
}
