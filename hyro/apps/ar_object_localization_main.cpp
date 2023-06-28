#include <hyro/common/Time.h>
#include <hyro/core/StateMachine.h>

#include <hyro/factory/CommandFactory.h>

#include <google/protobuf/wrappers.pb.h>
#include <hyro/utils/StateMachineSpinner.h>
#include <hyro/utils/Signal.h>

#include <hyro/ar_object_localization_component.h>

using namespace hyro;
using namespace std::string_literals;

int
main (int argc, char ** argv)
{
  std::cout<<"USAGE:\nar_object_localization <input_image_url> <input_detections_url> <is_stream> <model_file> <template_file> <cam_calib>"<<std::endl;
  bool is_stream=false;
  if (argc>3)
    is_stream=(argv[3][0]=='1');
  StateMachine ar_obj_loc_sm(std::make_shared<ArObjectLocalizationComponent>("/ar_object_localization"_uri, argv[4], argv[5], argv[6], is_stream));

  std::string ar_obj_loc_config;
  std::string input_proto="auto";

  
  ar_obj_loc_config = "{"
    "inputs: {"
      "input_image: { protocol: '" + input_proto + "',  max_queue_size: 1 },"
      "input_detections: { protocol: '" + input_proto + "', max_queue_size: 1 }"
    "}"
  "}";

  ar_obj_loc_sm.init(ComponentConfiguration(ar_obj_loc_config));
  
  std::string endpoint= argv[1];
  std::string endpoint_det= argv[2];
  ar_obj_loc_sm.start();
  
   RuntimeConfiguration rconf("{ endpoint: '"+endpoint+"' }");
  rconf.setParameter("max_queue_size", 1);
  RuntimeConfiguration rconf_det("{ endpoint: '"+endpoint_det+"' }");
  rconf_det.setParameter("max_queue_size", 1);

  ar_obj_loc_sm.connect(URI("input_image"), rconf);
  ar_obj_loc_sm.connect(URI("input_detections"), rconf_det);

  ar_obj_loc_sm.check();

  CancellationTokenSource cancellation_token;

  StateMachineSpinner ar_obj_loc_spinner(ar_obj_loc_sm, cancellation_token, 10ms);
  
  // Let the system run until the user presses Ctrl-C
  Signal sig(cancellation_token);
  sig.pause();

  // Wait for component completion
  ar_obj_loc_spinner.wait();

  ar_obj_loc_sm.reset();

  return 0;
}
