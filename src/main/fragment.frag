#version 450 core

out vec4 color;

in vec4 simPos;

uniform float time;
uniform sampler1D density;

void main()
{

  //color = vec4(texture(density, simPos.zw).xy, 0, 1);
  color = texture(density, simPos.z).xxxx;
}
