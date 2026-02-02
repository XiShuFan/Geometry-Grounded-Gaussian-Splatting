/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr
 */

#include <projects/gaussianviewer/renderer/GaussianView.hpp>
#include <core/graphics/GUI.hpp>
#include <thread>
#include <boost/asio.hpp>
#include <rasterizer.h>
#include <imgui_internal.h>

// Define the types and sizes that make up the contents of each Gaussian
// in the trained model.
typedef sibr::Vector3f Pos;
template <int D>
struct SHs
{
	float shs[(D + 1) * (D + 1) * 3];
};
struct Scale
{
	float scale[3];
};
struct Rot
{
	float rot[4];
};

template <int D>
struct Axis
{
	float axis[D * 3];
};

template <int D>
struct Sharpness
{
	float sharpness[D];
};

template <int D>
struct SGs
{
	float sgs[D * 3];
};

template <int SHD, int SGD>
struct RichPoint
{
	Pos pos;
	float n[3];
	SHs<SHD> shs;
	float opacity;
	Scale scale;
	Rot rot;
	Axis<SGD> axis;
	Sharpness<SGD> sharpness;
	SGs<SGD> sgs;
	float filter;
};

template <int SHD>
struct RichPoint<SHD, 0>
{
	Pos pos;
	float n[3];
	SHs<SHD> shs;
	float opacity;
	Scale scale;
	Rot rot;
	float filter;
};

float sigmoid(const float m1)
{
	return 1.0f / (1.0f + exp(-m1));
}

float inverse_sigmoid(const float m1)
{
	return log(m1 / (1.0f - m1));
}

float softplus(const float m1)
{
	return m1 > 20.f ? m1 : log(1.0f + exp(m1));
}

float inverse_softplus(const float m1)
{
	return m1 > 20.f ? m1 : log(exp(m1) - 1.0f);
}

template <class T>
T max(T a, T b)
{
	return a > b ? a : b;
}

#define CUDA_SAFE_CALL_ALWAYS(A)              \
	A;                                        \
	cudaDeviceSynchronize();                  \
	if (cudaPeekAtLastError() != cudaSuccess) \
		SIBR_ERR << cudaGetErrorString(cudaGetLastError());

#if DEBUG || _DEBUG
#define CUDA_SAFE_CALL(A) CUDA_SAFE_CALL_ALWAYS(A)
#else
#define CUDA_SAFE_CALL(A) A
#endif

// Load the Gaussians from the given file.
template <int SHD, int SGD>
int loadPly(const char *filename,
			std::vector<Pos> &pos,
			std::vector<float> &shs,
			std::vector<float> &opacities,
			std::vector<Scale> &scales,
			std::vector<Rot> &rot,
			std::vector<float> &axis,
			std::vector<float> &sharpness,
			std::vector<float> &sgs,
			sibr::Vector3f &minn,
			sibr::Vector3f &maxx)
{
	std::ifstream infile(filename, std::ios_base::binary);

	if (!infile.good())
		SIBR_ERR << "Unable to find model's PLY file, attempted:\n"
				 << filename << std::endl;

	// "Parse" header (it has to be a specific format anyway)
	std::string buff;
	std::getline(infile, buff);
	std::getline(infile, buff);

	std::string dummy;
	std::getline(infile, buff);
	std::stringstream ss(buff);
	int count;
	ss >> dummy >> dummy >> count;

	// Output number of Gaussians contained
	SIBR_LOG << "Loading " << count << " Gaussian splats" << std::endl;

	while (std::getline(infile, buff))
		if (buff.compare("end_header") == 0)
			break;

	// Read all Gaussians at once (AoS)
	std::vector<RichPoint<SHD, SGD>> points(count);
	infile.read((char *)points.data(), count * sizeof(RichPoint<SHD, SGD>));

	constexpr int SH_N = (SHD + 1) * (SHD + 1);
	// Resize our SoA data
	pos.resize(count);
	shs.resize(count * SH_N * 3);
	scales.resize(count);
	rot.resize(count);
	axis.resize(count * SGD * 3);
	sharpness.resize(count * SGD);
	sgs.resize(count * SGD * 3);
	opacities.resize(count);

	// Gaussians are done training, they won't move anymore. Arrange
	// them according to 3D Morton order. This means better cache
	// behavior for reading Gaussians that end up in the same tile
	// (close in 3D --> close in 2D).
	minn = sibr::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);
	maxx = -minn;
	for (int i = 0; i < count; i++)
	{
		maxx = maxx.cwiseMax(points[i].pos);
		minn = minn.cwiseMin(points[i].pos);
	}
	std::vector<std::pair<uint64_t, int>> mapp(count);
	for (int i = 0; i < count; i++)
	{
		sibr::Vector3f rel = (points[i].pos - minn).array() / (maxx - minn).array();
		sibr::Vector3f scaled = ((float((1 << 21) - 1)) * rel);
		sibr::Vector3i xyz = scaled.cast<int>();

		uint64_t code = 0;
		for (int i = 0; i < 21; i++)
		{
			code |= ((uint64_t(xyz.x() & (1 << i))) << (2 * i + 0));
			code |= ((uint64_t(xyz.y() & (1 << i))) << (2 * i + 1));
			code |= ((uint64_t(xyz.z() & (1 << i))) << (2 * i + 2));
		}

		mapp[i].first = code;
		mapp[i].second = i;
	}
	auto sorter = [](const std::pair<uint64_t, int> &a, const std::pair<uint64_t, int> &b)
	{
		return a.first < b.first;
	};
	std::sort(mapp.begin(), mapp.end(), sorter);

	// Move data from AoS to SoA
	for (int k = 0; k < count; k++)
	{
		int i = mapp[k].second;
		pos[k] = points[i].pos;

		// Normalize quaternion
		float length2 = 0;
		for (int j = 0; j < 4; j++)
			length2 += points[i].rot.rot[j] * points[i].rot.rot[j];
		float length = sqrt(length2);
		for (int j = 0; j < 4; j++)
			rot[k].rot[j] = points[i].rot.rot[j] / length;

		float det1 = 1;
		float det2 = 1;
		// Exponentiate scale
		for (int j = 0; j < 3; j++)
		{
			float scale_component = exp(points[i].scale.scale[j]);
			float filtered_scale = sqrt(scale_component * scale_component + points[i].filter * points[i].filter);
			det1 *= scale_component * scale_component;
			det2 *= scale_component * scale_component + points[i].filter * points[i].filter;
			scales[k].scale[j] = filtered_scale;
		}
		float coef = sqrt(det1 / det2);

		// Activate alpha
		opacities[k] = sigmoid(points[i].opacity) * coef;

		constexpr int SH_N3 = SH_N * 3;
		shs[k * SH_N3] = points[i].shs.shs[0];
		shs[k * SH_N3 + 1] = points[i].shs.shs[1];
		shs[k * SH_N3 + 2] = points[i].shs.shs[2];
		for (int j = 1; j < SH_N; j++)
		{
			shs[k * SH_N3 + j * 3 + 0] = points[i].shs.shs[(j - 1) + 3];
			shs[k * SH_N3 + j * 3 + 1] = points[i].shs.shs[(j - 1) + SH_N + 2];
			shs[k * SH_N3 + j * 3 + 2] = points[i].shs.shs[(j - 1) + 2 * SH_N + 1];
		}

		if constexpr (SGD > 0)
		{
			for (int j = 0; j < SGD; j++)
			{
				constexpr int SGD3 = SGD * 3;
				float length2 = 0;
				for (int d = 0; d < 3; d++)
					length2 += points[i].axis.axis[j * 3 + d] * points[i].axis.axis[j * 3 + d];
				float length = max(sqrt(length2), 1e-12f);
				for (int d = 0; d < 3; d++)
					axis[k * SGD3 + j * 3 + d] = points[i].axis.axis[j * 3 + d] / length;
				sharpness[k * SGD + j] = softplus(points[i].sharpness.sharpness[j]);
				for (int d = 0; d < 3; d++)
					sgs[k * SGD3 + j * 3 + d] = points[i].sgs.sgs[j * 3 + d];
			}
		}
	}
	return count;
}

template <int SHD, int SGD>
void savePly(const char *filename,
			 const std::vector<Pos> &pos,
			 const std::vector<float> &shs,
			 const std::vector<float> &opacities,
			 const std::vector<Scale> &scales,
			 const std::vector<Rot> &rot,
			 const std::vector<float> &axis,
			 const std::vector<float> &sharpness,
			 const std::vector<float> &sgs,
			 const sibr::Vector3f &minn,
			 const sibr::Vector3f &maxx)
{
	// Read all Gaussians at once (AoS)
	int count = 0;
	for (int i = 0; i < pos.size(); i++)
	{
		if (pos[i].x() < minn.x() || pos[i].y() < minn.y() || pos[i].z() < minn.z() ||
			pos[i].x() > maxx.x() || pos[i].y() > maxx.y() || pos[i].z() > maxx.z())
			continue;
		count++;
	}
	std::vector<RichPoint<SHD, SGD>> points(count);

	// Output number of Gaussians contained
	SIBR_LOG << "Saving " << count << " Gaussian splats" << std::endl;

	std::ofstream outfile(filename, std::ios_base::binary);

	outfile << "ply\nformat binary_little_endian 1.0\nelement vertex " << count << "\n";

	std::string props1[] = {"x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"};
	std::string props2[] = {"opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"};

	for (auto s : props1)
		outfile << "property float " << s << std::endl;
	for (int i = 0; i < ((SHD + 1) * (SHD + 1) - 1) * 3; i++)
		outfile << "property float f_rest_" << i << std::endl;
	for (auto s : props2)
		outfile << "property float " << s << std::endl;
	for (int i = 0; i < SGD * 3; i++)
		outfile << "property float sg_axis_" << i << std::endl;
	for (int i = 0; i < SGD; i++)
		outfile << "property float sg_sharpness_" << i << std::endl;
	for (int i = 0; i < SGD * 3; i++)
		outfile << "property float sg_color_" << i << std::endl;
	outfile << "property float filter_3D" << std::endl;
	outfile << "end_header" << std::endl;

	count = 0;
	for (int i = 0; i < pos.size(); i++)
	{
		if (pos[i].x() < minn.x() || pos[i].y() < minn.y() || pos[i].z() < minn.z() ||
			pos[i].x() > maxx.x() || pos[i].y() > maxx.y() || pos[i].z() > maxx.z())
			continue;
		points[count].pos = pos[i];
		points[count].rot = rot[i];
		// Exponentiate scale
		for (int j = 0; j < 3; j++)
			points[count].scale.scale[j] = log(scales[i].scale[j]);
		// Activate alpha
		points[count].opacity = inverse_sigmoid(opacities[i]);
		constexpr int SH_N = (SHD + 1) * (SHD + 1);
		constexpr int SH_N3 = SH_N * 3;
		constexpr int SGD3 = SGD * 3;
		points[count].shs.shs[0] = shs[i * SH_N3];
		points[count].shs.shs[1] = shs[i * SH_N3 + 1];
		points[count].shs.shs[2] = shs[i * SH_N3 + 2];
		for (int j = 1; j < SH_N; j++)
		{
			points[count].shs.shs[(j - 1) + 3] = shs[i * SH_N3 + j * 3];
			points[count].shs.shs[(j - 1) + 3 + SH_N - 1] = shs[i * SH_N3 + j * 3 + 1];
			points[count].shs.shs[(j - 1) + 3 + (SH_N - 1) * 2] = shs[i * SH_N3 + j * 3 + 2];
		}
		if constexpr (SGD > 0)
		{
			for (int j = 1; j < SGD; j++)
			{
				points[count].axis.axis[j * 3] = axis[i * SGD3 + j * 3];
				points[count].axis.axis[j * 3 + 1] = axis[i * SGD3 + j * 3 + 1];
				points[count].axis.axis[j * 3 + 2] = axis[i * SGD3 + j * 3 + 2];
				points[count].sharpness.sharpness[j] = inverse_softplus(sharpness[i * SGD + j]);
				points[count].sgs.sgs[j * 3] = sgs[i * SGD3 + j * 3];
				points[count].sgs.sgs[j * 3 + 1] = sgs[i * SGD3 + j * 3 + 1];
				points[count].sgs.sgs[j * 3 + 2] = sgs[i * SGD3 + j * 3 + 2];
			}
		}
		points[count].filter = 0.f;
		count++;
	}
	outfile.write((char *)points.data(), sizeof(RichPoint<SHD, SGD>) * points.size());
}

using LoaderFn = int (*)(const char *,
						 std::vector<Pos> &,
						 std::vector<float> &,
						 std::vector<float> &,
						 std::vector<Scale> &,
						 std::vector<Rot> &,
						 std::vector<float> &,
						 std::vector<float> &,
						 std::vector<float> &,
						 sibr::Vector3f &,
						 sibr::Vector3f &);

static constexpr LoaderFn kLoaders[4][8] = {
	{&loadPly<0, 0>, &loadPly<0, 1>, &loadPly<0, 2>, &loadPly<0, 3>,
	 &loadPly<0, 4>, &loadPly<0, 5>, &loadPly<0, 6>, &loadPly<0, 7>},
	{&loadPly<1, 0>, &loadPly<1, 1>, &loadPly<1, 2>, &loadPly<1, 3>,
	 &loadPly<1, 4>, &loadPly<1, 5>, &loadPly<1, 6>, &loadPly<1, 7>},
	{&loadPly<2, 0>, &loadPly<2, 1>, &loadPly<2, 2>, &loadPly<2, 3>,
	 &loadPly<2, 4>, &loadPly<2, 5>, &loadPly<2, 6>, &loadPly<2, 7>},
	{&loadPly<3, 0>, &loadPly<3, 1>, &loadPly<3, 2>, &loadPly<3, 3>,
	 &loadPly<3, 4>, &loadPly<3, 5>, &loadPly<3, 6>, &loadPly<3, 7>},
};

using SaverFn = void (*)(const char *,
						 const std::vector<Pos> &,
						 const std::vector<float> &,
						 const std::vector<float> &,
						 const std::vector<Scale> &,
						 const std::vector<Rot> &,
						 const std::vector<float> &,
						 const std::vector<float> &,
						 const std::vector<float> &,
						 const sibr::Vector3f &,
						 const sibr::Vector3f &);

static constexpr SaverFn kSavers[4][8] = {
	{&savePly<0, 0>, &savePly<0, 1>, &savePly<0, 2>, &savePly<0, 3>,
	 &savePly<0, 4>, &savePly<0, 5>, &savePly<0, 6>, &savePly<0, 7>},
	{&savePly<1, 0>, &savePly<1, 1>, &savePly<1, 2>, &savePly<1, 3>,
	 &savePly<1, 4>, &savePly<1, 5>, &savePly<1, 6>, &savePly<1, 7>},
	{&savePly<2, 0>, &savePly<2, 1>, &savePly<2, 2>, &savePly<2, 3>,
	 &savePly<2, 4>, &savePly<2, 5>, &savePly<2, 6>, &savePly<2, 7>},
	{&savePly<3, 0>, &savePly<3, 1>, &savePly<3, 2>, &savePly<3, 3>,
	 &savePly<3, 4>, &savePly<3, 5>, &savePly<3, 6>, &savePly<3, 7>},
};

namespace sibr
{
	// A simple copy renderer class. Much like the original, but this one
	// reads from a buffer instead of a texture and blits the result to
	// a render target.
	class BufferCopyRenderer
	{

	public:
		BufferCopyRenderer()
		{
			_shader.init("CopyShader",
						 sibr::loadFile(sibr::getShadersDirectory("gaussian") + "/copy.vert"),
						 sibr::loadFile(sibr::getShadersDirectory("gaussian") + "/copy.frag"));

			_flip.init(_shader, "flip");
			_width.init(_shader, "width");
			_height.init(_shader, "height");
		}

		void process(uint bufferID, IRenderTarget &dst, int width, int height, bool disableTest = true)
		{
			if (disableTest)
				glDisable(GL_DEPTH_TEST);
			else
				glEnable(GL_DEPTH_TEST);

			_shader.begin();
			_flip.send();
			_width.send();
			_height.send();

			dst.clear();
			dst.bind();

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufferID);

			sibr::RenderUtility::renderScreenQuad();

			dst.unbind();
			_shader.end();
		}

		/** \return option to flip the texture when copying. */
		bool &flip() { return _flip.get(); }
		int &width() { return _width.get(); }
		int &height() { return _height.get(); }

	private:
		GLShader _shader;
		GLuniform<bool> _flip = false; ///< Flip the texture when copying.
		GLuniform<int> _width = 1000;
		GLuniform<int> _height = 800;
	};
}

std::function<char *(size_t N)> resizeFunctional(void **ptr, size_t &S)
{
	auto lambda = [ptr, &S](size_t N)
	{
		if (N > S)
		{
			if (*ptr)
				CUDA_SAFE_CALL(cudaFree(*ptr));
			CUDA_SAFE_CALL(cudaMalloc(ptr, 2 * N));
			S = 2 * N;
		}
		return reinterpret_cast<char *>(*ptr);
	};
	return lambda;
}

sibr::GaussianView::GaussianView(const sibr::BasicIBRScene::Ptr &ibrScene, uint render_w, uint render_h, const char *file, bool *messageRead, int sh_degree, int sg_degree, bool white_bg, bool useInterop, int device) : _scene(ibrScene),
																																																						  _dontshow(messageRead),
																																																						  _sh_degree(sh_degree),
																																																						  _sg_degree(sg_degree),
																																																						  sibr::ViewBase(render_w, render_h)
{
	SIBR_LOG << "SH degree:" << sh_degree << '\t' << "SG degree:" << sg_degree << std::endl;
	int num_devices;
	CUDA_SAFE_CALL_ALWAYS(cudaGetDeviceCount(&num_devices));
	_device = device;
	if (device >= num_devices)
	{
		if (num_devices == 0)
			SIBR_ERR << "No CUDA devices detected!";
		else
			SIBR_ERR << "Provided device index exceeds number of available CUDA devices!";
	}
	CUDA_SAFE_CALL_ALWAYS(cudaSetDevice(device));
	cudaDeviceProp prop;
	CUDA_SAFE_CALL_ALWAYS(cudaGetDeviceProperties(&prop, device));
	if (prop.major < 7)
	{
		SIBR_ERR << "Sorry, need at least compute capability 7.0+!";
	}

	_pointbasedrenderer.reset(new PointBasedRenderer());
	_copyRenderer = new BufferCopyRenderer();
	_copyRenderer->flip() = true;
	_copyRenderer->width() = render_w;
	_copyRenderer->height() = render_h;

	std::vector<uint> imgs_ulr;
	const auto &cams = ibrScene->cameras()->inputCameras();
	for (size_t cid = 0; cid < cams.size(); ++cid)
	{
		if (cams[cid]->isActive())
		{
			imgs_ulr.push_back(uint(cid));
		}
	}
	_scene->cameras()->debugFlagCameraAsUsed(imgs_ulr);

	// Load the PLY data (AoS) to the GPU (SoA)
	std::cout << file << std::endl;
	std::vector<Pos> pos;
	std::vector<Rot> rot;
	std::vector<Scale> scale;
	std::vector<float> opacity;
	std::vector<float> shs;
	std::vector<float> axis;
	std::vector<float> sharpness;
	std::vector<float> sgs;

	count = kLoaders[sh_degree][sg_degree](file, pos, shs, opacity, scale, rot, axis, sharpness, sgs, _scenemin, _scenemax);

	_boxmin = _scenemin;
	_boxmax = _scenemax;

	int P = count;

	SIBR_LOG << "load success " << std::endl;

	const int SH_N = (sh_degree + 1) * (sh_degree + 1);
	// Allocate and fill the GPU data
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&pos_cuda, sizeof(Pos) * P));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(pos_cuda, pos.data(), sizeof(Pos) * P, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&rot_cuda, sizeof(Rot) * P));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(rot_cuda, rot.data(), sizeof(Rot) * P, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&shs_cuda, sizeof(float) * SH_N * 3 * P));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(shs_cuda, shs.data(), sizeof(float) * SH_N * 3 * P, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&opacity_cuda, sizeof(float) * P));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(opacity_cuda, opacity.data(), sizeof(float) * P, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&scale_cuda, sizeof(Scale) * P));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(scale_cuda, scale.data(), sizeof(Scale) * P, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&axis_cuda, sizeof(float) * max(sg_degree * 3 * P, 1)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&sharpness_cuda, sizeof(float) * max(sg_degree * P, 1)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&sgs_cuda, sizeof(float) * max(sg_degree * 3 * P, 1)));
	if (sg_degree > 0)
	{
		CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(axis_cuda, axis.data(), sizeof(float) * sg_degree * 3 * P, cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(sharpness_cuda, sharpness.data(), sizeof(float) * sg_degree * P, cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(sgs_cuda, sgs.data(), sizeof(float) * sg_degree * 3 * P, cudaMemcpyHostToDevice));
	}
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&mask_cuda, sizeof(float) * render_w * render_h));

	SIBR_LOG << "copy success " << std::endl;

	// Create space for view parameters
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&view_cuda, sizeof(sibr::Matrix4f)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&proj_cuda, sizeof(sibr::Matrix4f)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&cam_pos_cuda, 3 * sizeof(float)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&background_cuda, 3 * sizeof(float)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void **)&rect_cuda, 2 * P * sizeof(int)));

	float bg[3] = {white_bg ? 1.f : 0.f, white_bg ? 1.f : 0.f, white_bg ? 1.f : 0.f};
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(background_cuda, bg, 3 * sizeof(float), cudaMemcpyHostToDevice));

	gData = new GaussianData(P,
							 (float *)pos.data(),
							 (float *)rot.data(),
							 (float *)scale.data(),
							 opacity.data(),
							 (float *)shs.data(),
							 sh_degree);

	_gaussianRenderer = new GaussianSurfaceRenderer();

	// Create GL buffer ready for CUDA/GL interop
	glCreateBuffers(1, &imageBuffer);
	glNamedBufferStorage(imageBuffer, render_w * render_h * 3 * sizeof(float), nullptr, GL_DYNAMIC_STORAGE_BIT);

	if (useInterop)
	{
		if (cudaPeekAtLastError() != cudaSuccess)
		{
			SIBR_ERR << "A CUDA error occurred in setup:" << cudaGetErrorString(cudaGetLastError()) << ". Please rerun in Debug to find the exact line!";
		}
		cudaGraphicsGLRegisterBuffer(&imageBufferCuda, imageBuffer, cudaGraphicsRegisterFlagsWriteDiscard);
		useInterop &= (cudaGetLastError() == cudaSuccess);
	}
	if (!useInterop)
	{
		fallback_bytes.resize(render_w * render_h * 3 * sizeof(float));
		cudaMalloc(&fallbackBufferCuda, fallback_bytes.size());
		_interop_failed = true;
	}

	geomBufferFunc = resizeFunctional(&geomPtr, allocdGeom);
	binningBufferFunc = resizeFunctional(&binningPtr, allocdBinning);
	imgBufferFunc = resizeFunctional(&imgPtr, allocdImg);
	tileBufferFunc = resizeFunctional(&tilePtr, allocdTile);
}

void sibr::GaussianView::setScene(const sibr::BasicIBRScene::Ptr &newScene)
{
	_scene = newScene;

	// Tell the scene we are a priori using all active cameras.
	std::vector<uint> imgs_ulr;
	const auto &cams = newScene->cameras()->inputCameras();
	for (size_t cid = 0; cid < cams.size(); ++cid)
	{
		if (cams[cid]->isActive())
		{
			imgs_ulr.push_back(uint(cid));
		}
	}
	_scene->cameras()->debugFlagCameraAsUsed(imgs_ulr);
}

void sibr::GaussianView::onRenderIBR(sibr::IRenderTarget &dst, const sibr::Camera &eye)
{
	if (currMode == "Ellipsoids")
	{
		_gaussianRenderer->process(count, *gData, eye, dst, 0.2f, (_sh_degree + 1) * (_sh_degree + 1) * 3);
	}
	else if (currMode == "Initial Points")
	{
		_pointbasedrenderer->process(_scene->proxies()->proxy(), eye, dst);
	}
	else
	{
		// Convert view and projection to target coordinate system
		auto view_mat = eye.view();
		auto proj_mat = eye.viewproj();
		view_mat.row(1) *= -1;
		view_mat.row(2) *= -1;
		proj_mat.row(1) *= -1;

		// Compute additional view parameters
		float tan_fovy = tan(eye.fovy() * 0.5f);
		float tan_fovx = tan_fovy * eye.aspect();

		// Copy frame-dependent data to GPU
		CUDA_SAFE_CALL(cudaMemcpy(view_cuda, view_mat.data(), sizeof(sibr::Matrix4f), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(proj_cuda, proj_mat.data(), sizeof(sibr::Matrix4f), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(cam_pos_cuda, &eye.position(), sizeof(float) * 3, cudaMemcpyHostToDevice));

		float *image_cuda = nullptr;
		if (!_interop_failed)
		{
			// Map OpenGL buffer resource for use with CUDA
			size_t bytes;
			CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &imageBufferCuda));
			CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void **)&image_cuda, &bytes, imageBufferCuda));
		}
		else
		{
			image_cuda = fallbackBufferCuda;
		}

		// Rasterize
		// int* rects = _fastCulling ? rect_cuda : nullptr;
		int *rects = nullptr;
		float *boxmin = _cropping ? (float *)&_boxmin : nullptr;
		float *boxmax = _cropping ? (float *)&_boxmax : nullptr;
		CudaRasterizer::Rasterizer::forward(
			geomBufferFunc,
			binningBufferFunc,
			imgBufferFunc,
			tileBufferFunc,
			count, _sh_degree, (_sh_degree + 1) * (_sh_degree + 1), _sg_degree, _sg_degree,
			background_cuda,
			_resolution.x(), _resolution.y(),
			pos_cuda,
			nullptr,
			opacity_cuda,
			scale_cuda,
			rot_cuda,
			nullptr,
			shs_cuda,
			axis_cuda,
			sharpness_cuda,
			sgs_cuda,
			_scalingModifier,
			view_cuda,
			proj_cuda,
			cam_pos_cuda,
			tan_fovx,
			tan_fovy,
			0,
			false,
			image_cuda,
			nullptr,
			mask_cuda,
			nullptr,
			nullptr,
			false,
			false);

		if (!_interop_failed)
		{
			// Unmap OpenGL resource for use with OpenGL
			CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &imageBufferCuda));
		}
		else
		{
			CUDA_SAFE_CALL(cudaMemcpy(fallback_bytes.data(), fallbackBufferCuda, fallback_bytes.size(), cudaMemcpyDeviceToHost));
			glNamedBufferSubData(imageBuffer, 0, fallback_bytes.size(), fallback_bytes.data());
		}
		// Copy image contents to framebuffer
		_copyRenderer->process(imageBuffer, dst, _resolution.x(), _resolution.y());
	}

	if (cudaPeekAtLastError() != cudaSuccess)
	{
		SIBR_ERR << "A CUDA error occurred during rendering:" << cudaGetErrorString(cudaGetLastError()) << ". Please rerun in Debug to find the exact line!";
	}
}

void sibr::GaussianView::onUpdate(Input &input)
{
}

void sibr::GaussianView::onGUI()
{
	// Generate and update UI elements
	const std::string guiName = "3D Gaussians";
	if (ImGui::Begin(guiName.c_str()))
	{
		if (ImGui::BeginCombo("Render Mode", currMode.c_str()))
		{
			if (ImGui::Selectable("Splats"))
				currMode = "Splats";
			if (ImGui::Selectable("Initial Points"))
				currMode = "Initial Points";
			if (ImGui::Selectable("Ellipsoids"))
				currMode = "Ellipsoids";
			ImGui::EndCombo();
		}
	}
	if (currMode == "Splats")
	{
		ImGui::SliderFloat("Scaling Modifier", &_scalingModifier, 0.001f, 1.0f);
	}
	ImGui::Checkbox("Fast culling", &_fastCulling);

	ImGui::Checkbox("Crop Box", &_cropping);
	if (_cropping)
	{
		ImGui::SliderFloat("Box Min X", &_boxmin.x(), _scenemin.x(), _scenemax.x());
		ImGui::SliderFloat("Box Min Y", &_boxmin.y(), _scenemin.y(), _scenemax.y());
		ImGui::SliderFloat("Box Min Z", &_boxmin.z(), _scenemin.z(), _scenemax.z());
		ImGui::SliderFloat("Box Max X", &_boxmax.x(), _scenemin.x(), _scenemax.x());
		ImGui::SliderFloat("Box Max Y", &_boxmax.y(), _scenemin.y(), _scenemax.y());
		ImGui::SliderFloat("Box Max Z", &_boxmax.z(), _scenemin.z(), _scenemax.z());
		ImGui::InputText("File", _buff, 512);
		if (ImGui::Button("Save"))
		{
			int sh_dim = (_sh_degree + 1) * (_sh_degree + 1) * 3;
			std::vector<Pos> pos(count);
			std::vector<Rot> rot(count);
			std::vector<float> opacity(count);
			std::vector<float> shs(count * sh_dim);
			std::vector<Scale> scale(count);
			std::vector<float> axis(count * _sg_degree * 3);
			std::vector<float> sharpness(count * _sg_degree);
			std::vector<float> sgs(count * _sg_degree * 3);
			CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(pos.data(), pos_cuda, sizeof(Pos) * count, cudaMemcpyDeviceToHost));
			CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(rot.data(), rot_cuda, sizeof(Rot) * count, cudaMemcpyDeviceToHost));
			CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(opacity.data(), opacity_cuda, sizeof(float) * count, cudaMemcpyDeviceToHost));
			CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(shs.data(), shs_cuda, sizeof(float) * count * sh_dim, cudaMemcpyDeviceToHost));
			CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(scale.data(), scale_cuda, sizeof(Scale) * count, cudaMemcpyDeviceToHost));
			CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(axis.data(), axis_cuda, sizeof(float) * count * _sg_degree * 3, cudaMemcpyDeviceToHost));
			CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(sharpness.data(), sharpness_cuda, sizeof(float) * count * _sg_degree, cudaMemcpyDeviceToHost));
			CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(sgs.data(), sgs_cuda, sizeof(float) * count * _sg_degree * 3, cudaMemcpyDeviceToHost));
			kSavers[_sh_degree][_sg_degree](_buff, pos, shs, opacity, scale, rot, axis, sharpness, sgs, _boxmin, _boxmax);
		}
	}

	ImGui::End();

	if (!*_dontshow && !accepted && _interop_failed)
		ImGui::OpenPopup("Error Using Interop");

	if (!*_dontshow && !accepted && _interop_failed && ImGui::BeginPopupModal("Error Using Interop", NULL, ImGuiWindowFlags_AlwaysAutoResize))
	{
		ImGui::SetItemDefaultFocus();
		ImGui::SetWindowFontScale(2.0f);
		ImGui::Text("This application tries to use CUDA/OpenGL interop.\n"
					" It did NOT work for your current configuration.\n"
					" For highest performance, OpenGL and CUDA must run on the same\n"
					" GPU on an OS that supports interop.You can try to pass a\n"
					" non-zero index via --device on a multi-GPU system, and/or try\n"
					" attaching the monitors to the main CUDA card.\n"
					" On a laptop with one integrated and one dedicated GPU, you can try\n"
					" to set the preferred GPU via your operating system.\n\n"
					" FALLING BACK TO SLOWER RENDERING WITH CPU ROUNDTRIP\n");

		ImGui::Separator();

		if (ImGui::Button("  OK  "))
		{
			ImGui::CloseCurrentPopup();
			accepted = true;
		}
		ImGui::SameLine();
		ImGui::Checkbox("Don't show this message again", _dontshow);
		ImGui::EndPopup();
	}
}

sibr::GaussianView::~GaussianView()
{
	// Cleanup
	cudaFree(pos_cuda);
	cudaFree(rot_cuda);
	cudaFree(scale_cuda);
	cudaFree(opacity_cuda);
	cudaFree(shs_cuda);
	cudaFree(axis_cuda);
	cudaFree(sharpness_cuda);
	cudaFree(sgs_cuda);

	cudaFree(view_cuda);
	cudaFree(proj_cuda);
	cudaFree(cam_pos_cuda);
	cudaFree(background_cuda);
	cudaFree(rect_cuda);

	if (!_interop_failed)
	{
		cudaGraphicsUnregisterResource(imageBufferCuda);
	}
	else
	{
		cudaFree(fallbackBufferCuda);
	}
	glDeleteBuffers(1, &imageBuffer);

	if (geomPtr)
		cudaFree(geomPtr);
	if (binningPtr)
		cudaFree(binningPtr);
	if (imgPtr)
		cudaFree(imgPtr);

	delete _copyRenderer;
}
