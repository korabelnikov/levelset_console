#include "LevelsetOpenCL.h"

#include <agtkIO.h>

#include "itkImage.h"

#include <chrono>

int main(int argc, char *argv[])
{

  const std::string imageFilename = "B:/patient.nrrd", outputFilename = "B:/patient_Resampled_1_1_1_out.nrrd";

  ImageType::Pointer image = ImageType::New();
  agtk::readImage(image, imageFilename);
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

  agtk::FloatImage3D::Pointer levelset = agtk::FloatImage3D::New();
  levelset->CopyInformation(image);
  levelset->SetRegions(image->GetLargestPossibleRegion());
  levelset->Allocate();
  levelset->FillBuffer(0);

  auto m_Levelset = new LevelSetOpenCL(image, output);
  const int radius = 5;
  printExecTime(
    m_Levelset->initializeLevelSet(new int[4]{center[0], center[1], center[2]}, radius);
  );

  const int itersLS = 120;
  const int itersNB = 10;
  printExecTime(
    m_Levelset->runLevelSet(itersLS, itersNB, threshold, epsilon, alpha);
  );
/*
  m_Levelset->copyToItk(m_Levelset->m_Levelset, levelset);
  printExecTime(
    agtk::writeImage(levelset, "B:/levelset.nrrd");
  );
*/
  m_Levelset->thresholdLevelSet(0);
  m_Levelset->copyToItk(m_Levelset->m_LevelsetBinary, output);

  printExecTime(
    agtk::writeImage(output, outputFilename);
  );

  delete m_Levelset;

  getchar();

  return EXIT_SUCCESS;
}

//int main(int argc, char *argv[])
//{
//
//  const std::string imageFilename = "B:/0229.nrrd", outputFilename = "B:/0229seg.nrrd";
//
//  ImageType::Pointer image = ImageType::New();
//  agtk::readImage(image, imageFilename);
//  agtk::BinaryImage3D::OffsetType center = { 139, 225, 203 };
//
//
//  LevelSetNarrowBandCPU::Params params;
//  ImageType::IndexType centerInd = { center[0], center[1], center[2] };
//  params.threshold = image->GetPixel(centerInd);
//  params.epsilon = 90;
//  params.alpha = 0.04;
//
//  auto m_Levelset = new LevelSetNarrowBandCPU(image);
//  const int radius = 4;
//  printExecTime(
//    m_Levelset->initialize(center, radius);
//  );
//
//  m_Levelset->setParams(params);
//
//  const int iters = 500;
//  printExecTime(
//    m_Levelset->updateLevelSetFunction(iters);
//  );
//  
//  printExecTime(
//  agtk::writeImage(m_Levelset->getLevelset(), outputFilename);
//  );
//
//  delete m_Levelset;
//
//  getchar();
//
//  return EXIT_SUCCESS;
//}
