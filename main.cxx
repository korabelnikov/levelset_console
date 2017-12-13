#include "LevelsetOpenCL.h"

#include <agtkIO.h>

#include "itkImage.h"

#include <chrono>

#define IMAGE_DIR "C:/images/"

class OpenCLDebugger
{
public:
  static void run()
  {
    const std::string imageFilename = std::string(IMAGE_DIR) + "patient.nrrd",
      outputFilename = std::string(IMAGE_DIR) + "patient_out.nrrd";

    ImageType::Pointer image = ImageType::New();
    agtk::readImage<ImageType>(image, imageFilename);
    agtk::BinaryImage3D::OffsetType center = { 185, 300, 65 };// { 100,170,100 };

    LevelSetOpenCL::staticConstructor();

    ImageType::IndexType centerInd = { center[0], center[1], center[2] };
    float threshold = 110;// image->GetPixel(centerInd);
    float epsilon = 80;
    float alpha = 0.010;

    agtk::BinaryImage3D::Pointer output = agtk::BinaryImage3D::New();
    output->CopyInformation(image);
    output->SetRegions(image->GetLargestPossibleRegion());
    output->Allocate();
    output->FillBuffer(0);

    ImageType::Pointer levelset = ImageType::New();
    levelset->CopyInformation(image);
    levelset->SetRegions(image->GetLargestPossibleRegion());
    levelset->Allocate();
    levelset->FillBuffer(0);

    auto m_Levelset = new LevelSetOpenCL(image, output, output);
    m_Levelset->updateInput();

    m_Levelset->getModifiedParams().m_Alpha = alpha;
    m_Levelset->getModifiedParams().m_Eps = epsilon;
    m_Levelset->getModifiedParams().m_Threshold = threshold;

    printExecTime(
      m_Levelset->reinitialize(new int[4]{ center[0], center[1], center[2] });
      m_Levelset->beginPrefetching();
      //todo debug
      m_Levelset->m_PrefethcingFuture->wait();
    );

    const int iters = 2000;
    printExecTime(
      int n = 1;
    for (int i = 0; i < n; ++i)
      m_Levelset->moveLevelset(iters / n);
    );

    m_Levelset->stopPrefetching();
    m_Levelset->fixFullSizedOutput(true);

    m_Levelset->copyToHost(m_Levelset->m_LevelsetBinaryAccum, levelset->GetBufferPointer());
    agtk::writeImage<ImageType>(levelset, "C:/images/levelset_accum.nrrd");

    m_Levelset->copyToHost(m_Levelset->m_Levelset, levelset->GetBufferPointer());
    agtk::writeImage<ImageType>(levelset, "C:/images/levelset.nrrd");

    printExecTime(
      agtk::writeImage(output, outputFilename);
    );

    //printExecTime(
    //  m_Levelset->moveLevelset(-20);
    //);
    //m_Levelset->copyToHost(m_Levelset->getLevelsetBinary(), output->GetBufferPointer());
    //agtk::writeImage(output, "C:/images/levelset_out-20.nrrd");
    ////
    //printExecTime(
    //  m_Levelset->moveLevelset(-500);
    //);
    //m_Levelset->copyToHost(m_Levelset->getLevelsetBinary(), output->GetBufferPointer());
    //agtk::writeImage(output, "C:/images/levelset_out-700.nrrd");
    ////

    printExecTime(
      m_Levelset->blurFullSizedOutput();
      );

    m_Levelset->copyToHost(m_Levelset->m_FullSizedOutput, output->GetBufferPointer());
    agtk::writeImage(output, "C:/images/m_FullSizedOutput_blur.nrrd");

  }

};

int main(int argc, char *argv[])
{
  OpenCLDebugger::run();

  system("pause");

  return EXIT_SUCCESS;
}
