#include "LevelsetOpenCL.h"

#include <agtkIO.h>

#include "itkImage.h"

#include <chrono>

#define IMAGE_DIR "C:/images/"

int main(int argc, char *argv[])
{

  const std::string imageFilename = std::string(IMAGE_DIR) + "patient.nrrd",
  outputFilename = std::string(IMAGE_DIR) + "patient_out.nrrd";

  ImageType::Pointer image = ImageType::New();
  agtk::readImage<ImageType>(image, imageFilename);
  agtk::BinaryImage3D::OffsetType center = { 185, 300, 65 };// { 100,170,100 };

  LevelSetOpenCL::staticConstructor();

  ImageType::IndexType centerInd = { center[0], center[1], center[2] };
  float threshold = 110;// image->GetPixel(centerInd);
  float epsilon = 120;
  float alpha = 0.04;

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

  const int radius = 5;
  printExecTime(
    m_Levelset->initializeLevelSet(new int[4]{center[0], center[1], center[2]}, radius);
  );

  const int itersLS = 120;
  const int itersNB = 10;
  printExecTime(
    m_Levelset->runLevelSet(itersLS, itersNB, threshold, epsilon, alpha);
  );

  m_Levelset->copyToItk(m_Levelset->getLevelset(), levelset);
  printExecTime(
    agtk::writeImage<ImageType>(levelset, std::string(IMAGE_DIR) + "levelset.nrrd");
  );

  m_Levelset->thresholdLevelSet(0);
  m_Levelset->copyToItk(m_Levelset->getLevelsetBinary(), output);

  printExecTime(
    agtk::writeImage(output, outputFilename);
  );

  delete m_Levelset;

  getchar();

  return EXIT_SUCCESS;
}
