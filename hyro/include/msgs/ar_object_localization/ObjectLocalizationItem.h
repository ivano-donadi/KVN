#pragma once

#ifndef HYRO_MSGS_AR_OBJECT_LOCALIZATION_ITEM_H
#define HYRO_MSGS_AR_OBJECT_LOCALIZATION_ITEM_H

#include <hyro/msgs/common/Basic.h>
#include <hyro/msgs/common/Header.h>
#include <hyro/msgs/navigation/Pose.h>
#include <string>

namespace hyro
{
// struct AR_OBJECT_DETECTION_ITEM_MSGS_EXPORT ObjectLocalizationItem
struct ObjectDetectionItem
{ 
  std::string object_name;
  Pose pose;
  float confidence;
};

struct ObjectLocalization
{
  Header header;
  std::vector<ObjectLoacalizationItem> localizations;
};

inline std::ostream &
operator<<(std::ostream &os,
           const ObjectLocalizationItem &item)
{
  os << "{object_name: " << item.object_name << ",\n";
  os << "pose: " << item.pose << ",\n";
  os << "confidence: " << item.confidence << "}\n";
  return os;
}
} // namespace hyro


#include <hyro/msgs/ar_object_localization/ObjectLocalizationItem.proto.h>

#endif // HYRO_MSGS_AR_OBJECT_LOCALIZATION_ITEM_H
