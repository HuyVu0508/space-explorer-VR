using UnityEngine;
using UnityEngine.AI;
using UnityEngine.EventSystems;
using UnityEngine.SceneManagement;



namespace ControllerSelection {
    public class teleport_scene : MonoBehaviour
    {

        public GameObject teleport_canvas;
        private bool canvas_enabled;

        // Start is called before the first frame update
        void Start()
        {
            teleport_canvas.SetActive(false);
            canvas_enabled = false;            
        }

        // Update is called once per frame
        void Update()
        {
            TurnOnOffTeleportCanvas();
        }

        // Changing scence
        public void OnChangeScene(){

            string button_name = EventSystem.current.currentSelectedGameObject.name;

            // Loading level with scene name
            if (button_name == "spaceshipscene"){
                SceneManager.LoadScene("SpaceshipScene");
            }
            if (button_name == "cargoscene"){
                SceneManager.LoadScene("CargoScene");
            }       
            if (button_name == "toolsroomscene"){
                SceneManager.LoadScene("ToolsRoomScene");
            }       
            if (button_name == "greenhousescene"){
                SceneManager.LoadScene("GreenHouseScene");
            }                                    
        }        

        // Switch on off canvas
        public void TurnOnOffTeleportCanvas(){
            if (Input.GetKeyDown(KeyCode.T) || OVRInput.GetUp(OVRInput.Button.Four)){

                if (canvas_enabled){
                    teleport_canvas.SetActive(false);
                    canvas_enabled = false;
                }
                else{
                    teleport_canvas.SetActive(true);
                    canvas_enabled = true;            
                }
            }            
        }

    }
}
