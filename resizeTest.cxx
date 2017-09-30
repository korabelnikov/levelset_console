#include "LevelsetOpenCL.h"

#include <agtkIO.h>

#include "itkImage.h"

#include <chrono>
#include <functional>

int main(int argc, char *argv[])
{
  LevelSetOpenCL::staticConstructor();

  //sizes
  agtk::Image3DSize m_Size9x, m_Size3x, m_Size1x;
  m_Size9x.Fill(10);

  m_Size3x[0] = m_Size3x[1] = m_Size9x[0] * 3;
  m_Size3x[2] = m_Size9x[0];

  m_Size1x[0] = m_Size1x[1] = m_Size3x[0] * 3;
  m_Size1x[2] = m_Size9x[0];

   //spacings
  agtk::Image3DSpacing spacing9x, spacing3x, spacing1x;
  spacing9x.Fill(1);

  spacing3x[0] = spacing3x[1] = spacing9x[0] / 3;
  spacing3x[2] = 1;

  spacing1x[0] = spacing1x[1] = spacing3x[0] / 3;
  spacing1x[2] = 1;

  agtk::FloatImage3D::Pointer input9x = agtk::FloatImage3D::New();
  input9x->SetSpacing(spacing9x);
  input9x->SetRegions(agtk::Image3DRegion(m_Size9x));
  input9x->Allocate();
  input9x->FillBuffer(0);
  agtk::Image3DIndex index;
  index.Fill(1);
  input9x->SetPixel(index, 1);
  //save
  agtk::writeImage(input9x, std::string("B:/input9x.nrrd"));
 
  auto m_QueueExec = cl::CommandQueue(LevelSetOpenCL::s_OpenCL->context, LevelSetOpenCL::s_OpenCL->device);

  const int Dim = 3;
  cl::size_t<Dim> m_OriginZero; //todo const
  for (int i = 0; i < Dim; ++i) {
    m_OriginZero[i] = 0;
  }

  auto copyToItk = [&](cl::Image3D from, ImageType* to, const cl::size_t<Dim>& size)
  {
    m_QueueExec.enqueueReadImage(
      from,
      CL_TRUE,
      m_OriginZero,
      size,
      0, 0,
      to->GetBufferPointer()
      );
    m_QueueExec.finish();
  };

  auto downsample = [&](cl::Image3D from, cl::Image3D to, agtk::Image3DSize size) {
    cl::Kernel downsample3Kernel(LevelSetOpenCL::s_OpenCL->program, "downsample3x3");
    downsample3Kernel.setArg(0, from);
    downsample3Kernel.setArg(1, to);
    m_QueueExec.enqueueNDRangeKernel(
      downsample3Kernel,
      cl::NullRange,
      cl::NDRange(size[0], size[1], size[2]),
      cl::NullRange
      );

    m_QueueExec.finish();
  };

  auto upsample = [&](cl::Image3D from, cl::Image3D to, agtk::Image3DSize size) {
    cl::Kernel downsample3Kernel(LevelSetOpenCL::s_OpenCL->program, "upsample3_1");
    downsample3Kernel.setArg(0, from);
    downsample3Kernel.setArg(1, to);
    m_QueueExec.enqueueNDRangeKernel(
      downsample3Kernel,
      cl::NullRange,
      cl::NDRange(size[0], size[1], size[2]),
      cl::NullRange
      );

    m_QueueExec.finish();
  };

  auto save = [&](cl::Image3D image, agtk::Image3DSize size, agtk::Image3DSpacing spacing, std::string filename) {
    auto imageItk = agtk::FloatImage3D::New();
    imageItk->SetRegions(agtk::Image3DRegion(size));
    imageItk->SetSpacing(spacing);
    imageItk->Allocate();
    imageItk->FillBuffer(0);

    cl::size_t<3> sizeCl;
    sizeCl[0] = size[0];
    sizeCl[1] = size[1];
    sizeCl[2] = size[2];
    copyToItk(image, imageItk, sizeCl);
    agtk::writeImage(imageItk, filename);
  };

  auto m_InputData9x = cl::Image3D(
    LevelSetOpenCL::s_OpenCL->context,
    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
    cl::ImageFormat(CL_R, CL_FLOAT),
    m_Size9x[0], // Width(),
    m_Size9x[1], // Height(),
    m_Size9x[2], // Depth(),
    0, 0,
    const_cast<float*>(input9x->GetBufferPointer())
    );
  auto m_InputData9x_ = cl::Image3D(
    LevelSetOpenCL::s_OpenCL->context,
    CL_MEM_READ_WRITE,
    cl::ImageFormat(CL_R, CL_FLOAT),
    m_Size9x[0], // Width(),
    m_Size9x[1], // Height(),
    m_Size9x[2] // Depth(),
    );

  auto m_InputData3x_ = cl::Image3D(
    LevelSetOpenCL::s_OpenCL->context,
    CL_MEM_READ_WRITE,
    cl::ImageFormat(CL_R, CL_FLOAT),
    m_Size3x[0], // Width(),
    m_Size3x[1], // Height(),
    m_Size3x[2] // Depth(),
    );

  auto m_InputData3x = cl::Image3D(
    LevelSetOpenCL::s_OpenCL->context,
    CL_MEM_READ_WRITE,
    cl::ImageFormat(CL_R, CL_FLOAT),
    m_Size3x[0], // Width(),
    m_Size3x[1], // Height(),
    m_Size3x[2] // Depth(),
    );

  auto m_InputData1x = cl::Image3D(
    LevelSetOpenCL::s_OpenCL->context,
    CL_MEM_READ_WRITE,
    cl::ImageFormat(CL_R, CL_FLOAT),
    m_Size1x[0], // Width(),
    m_Size1x[1], // Height(),
    m_Size1x[2] // Depth(),
    );

  //proc
  upsample(m_InputData9x, m_InputData3x, m_Size3x);
  save(m_InputData3x, m_Size3x, spacing3x, std::string("B:/input3x.nrrd"));

  upsample(m_InputData3x, m_InputData1x, m_Size1x);
  save(m_InputData1x, m_Size1x, spacing1x, std::string("B:/input1x.nrrd"));

  downsample(m_InputData1x, m_InputData3x_, m_Size3x);
  save(m_InputData3x_, m_Size3x, spacing3x, std::string("B:/input3x_.nrrd"));
 
  downsample(m_InputData3x, m_InputData9x_, m_Size9x);
  save(m_InputData9x_, m_Size9x, spacing9x, std::string("B:/input9x_.nrrd"));
 
  system("pause");

  return EXIT_SUCCESS;
}

