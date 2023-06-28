#pragma once

#ifndef HYRO_MSGS_AR_OBJECT_LOCALIZATION_ITEM_PROTO_H
#define HYRO_MSGS_AR_OBJECT_LOCALIZATION_ITEM_PROTO_H

#include <hyro/msgs/ar_object_localization/ObjectLocalizationItem.h>
#include <hyro/dynamic/ProtobufTraits.h>
#include <hyro/msgs/ar_object_localization/ObjectLocalizationItem.pb.h>

namespace hyro
{
  template <>
  struct ProtobufTraits<ObjectLocalizationItem> : public ProtobufTraitsDefault<ObjectLocalizationItem, msgs::ar_object_localization::ObjectLocalizationItem>
  {
    static void
    FromMessage (const msgs::ar_object_localization::ObjectLocalizationItem& msg,
                ObjectLocalizationItem * value)
    {
      value->object_name = msg.object_name();
      utils::FromMessage(msg.pose(), &value->pose);
      value->confidence = msg.confidence();
    }

    static void
    ToMessage (const ObjectLocalizationItem & value,
              msgs::ar_object_localization::ObjectLocalizationItem * msg)
    {
      msg->set_object_name(value.object_name);
      utils::ToMessage(value.pose, msg->mutable_pose());
      msg->set_confidence(value.confidence);
    }
  };
  
  
  template <>
  struct ProtobufTraits<ObjectLocalization> : public ProtobufTraitsDefault<ObjectLocalization, msgs::ar_object_localization::ObjectLocalization>
  {
    static void
    FromMessage (const msgs::ar_object_localization::ObjectLocalization& msg,
                ObjectLocalization * value)
    {
      utils::FromMessage(msg.header(), &value->header);
      utils::FromMessage(msg.localizations(), &value->localizations);
    }

    static void
    ToMessage (const ObjectLocalization & value,
              msgs::ar_object_localization::ObjectLocalization * msg)
    {
      utils::ToMessage(value.header, msg->mutable_header());
      utils::ToMessage(value.localizations, msg->mutable_localizations());
    }
  };
  
} // namespace hyro

#endif // HYRO_MSGS_AR_OBJECT_LOCALIZATION_ITEM_PROTO_H
