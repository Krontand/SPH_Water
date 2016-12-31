#version 330 core

uniform mat4 modelViewProjectionMatrix;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrixInverse;

in vec3 position;
in vec3 color;

out vec3 fragmentColor;

out mat4 VPMTInverse;
out mat4 VPInverse;
out vec3 centernormclip;

void main(void)
{
	int width = 1360;
	int height = 768;
	float R = 0.012;
	gl_Position = modelViewProjectionMatrix * vec4(position, 1.0);
	fragmentColor = color;

    mat4 T = mat4(
            1.0,			0.0,			0.0,			0.0,
            0.0,			1.0,			0.0,			0.0,
            0.0,			0.0,			1.0,			0.0,
            position.x/R,	position.y/R,	position.z/R,	1.0/R);

    mat4 PMTt = transpose(modelViewProjectionMatrix * T);

    vec4 r1 = PMTt[0];
    vec4 r2 = PMTt[1];
    vec4 r4 = PMTt[3];
    float r1Dr4T = dot(r1.xyz,r4.xyz)-r1.w*r4.w;
    float r1Dr1T = dot(r1.xyz,r1.xyz)-r1.w*r1.w;
    float r4Dr4T = dot(r4.xyz,r4.xyz)-r4.w*r4.w;
    float r2Dr2T = dot(r2.xyz,r2.xyz)-r2.w*r2.w;
    float r2Dr4T = dot(r2.xyz,r4.xyz)-r2.w*r4.w;

    gl_Position = vec4(-r1Dr4T, -r2Dr4T, gl_Position.z/gl_Position.w*(-r4Dr4T), -r4Dr4T);

    float discriminant_x = r1Dr4T*r1Dr4T-r4Dr4T*r1Dr1T;
    float discriminant_y = r2Dr4T*r2Dr4T-r4Dr4T*r2Dr2T;
    float screen = width;

    gl_PointSize = sqrt(max(discriminant_x, discriminant_y))*screen/(-r4Dr4T);


    // prepare varyings

    mat4 TInverse = mat4(
            1.0,          0.0,          0.0,         0.0,
            0.0,          1.0,          0.0,         0.0,
            0.0,          0.0,          1.0,         0.0,
            -position.x,  -position.y,  -position.z, R);
    mat4 VInverse = mat4(
            2.0/width, 0.0,        0.0,                    0.0,
            0.0,	   2.0/height, 0.0,                    0.0,
            0.0,       0.0,        2.0/gl_DepthRange.diff, 0.0,
            -1,        -1,         -(gl_DepthRange.near+gl_DepthRange.far)/gl_DepthRange.diff, 1.0);

	mat4 modelViewProjectionMatrixInverse = inverse(modelViewProjectionMatrix);

    VPMTInverse = TInverse* modelViewProjectionMatrixInverse*VInverse;
    VPInverse = projectionMatrixInverse*VInverse; // TODO: move to CPU
    vec4 centerclip = modelViewMatrix * vec4(position, 1.0);
    centernormclip = vec3(centerclip)/centerclip.w;
}
