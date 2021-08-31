using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class angle_computing : MonoBehaviour
{
    public UnityEngine.UI.Text outText;

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        Vector3 rotationL = OVRInput.GetLocalControllerRotation(OVRInput.Controller.LTouch).eulerAngles;
        Vector3 rotationR = OVRInput.GetLocalControllerRotation(OVRInput.Controller.RTouch).eulerAngles;

        string steer_direction = "middle";
        // if (((rotationL.y > 290) && (rotationL.y < 360)) && ((rotationR.y > 290) && (rotationR.y < 360))){
        if (((rotationL.y > 20) && (rotationL.y < 90)) && ((rotationR.y > 20) && (rotationR.y < 90))){
            steer_direction = "right";
        }
        if (((rotationL.y > 270) && (rotationL.y < 340)) && ((rotationR.y > 270) && (rotationR.y < 340))){
            steer_direction = "left";
        }

        // outText.text = rotationL.ToString() + " // " + rotationR.ToString() + " // " + steer_direction;
        // Debug.Log(steer_direction.ToString());



    }
}
