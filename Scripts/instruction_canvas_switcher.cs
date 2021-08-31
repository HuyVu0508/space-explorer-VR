using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class instruction_canvas_switcher : MonoBehaviour
{

    public GameObject instruction_canvas;
    private bool canvas_enabled;

    // Start is called before the first frame update
    void Start()
    {
        instruction_canvas.SetActive(false);
        canvas_enabled = false;           
    }

    // Update is called once per frame
    void Update()
    {
        TurnOnOffTeleportCanvas();
    }

    // Switch on off canvas
    public void TurnOnOffTeleportCanvas(){
        if (Input.GetKeyDown(KeyCode.Y) || OVRInput.GetUp(OVRInput.Button.Three)){

            if (canvas_enabled){
                instruction_canvas.SetActive(false);
                canvas_enabled = false;
            }
            else{
                instruction_canvas.SetActive(true);
                canvas_enabled = true;            
            }
        }            
    }    
}
