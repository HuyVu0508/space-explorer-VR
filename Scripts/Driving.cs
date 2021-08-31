using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Driving : MonoBehaviour
{
    public GameObject Spaceship;
    public GameObject SteeringWheel;
    public float rotate_speed;
    public float accelerate_speed;

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        // //  Rotation
        // if ((Input.GetKey(KeyCode.J) || OVRInput.Get(OVRInput.Button.PrimaryThumbstickLeft)) &&
        //     (Input.GetKey(KeyCode.J) || OVRInput.Get(OVRInput.Button.SecondaryThumbstickLeft)))
        // {
        //     Spaceship.transform.RotateAround(Spaceship.transform.position, Vector3.down, rotate_speed * Time.deltaTime);
        //     SteeringWheel.transform.RotateAround(SteeringWheel.transform.position, SteeringWheel.transform.forward, rotate_speed * 2 * Time.deltaTime);
        // }         
        // if ((Input.GetKey(KeyCode.L) || OVRInput.Get(OVRInput.Button.PrimaryThumbstickRight)) &&
        //     (Input.GetKey(KeyCode.L) || OVRInput.Get(OVRInput.Button.SecondaryThumbstickRight)))
        // {
        //     Spaceship.transform.RotateAround(Spaceship.transform.position, Vector3.down, - rotate_speed * Time.deltaTime);
        //     SteeringWheel.transform.RotateAround(SteeringWheel.transform.position, SteeringWheel.transform.forward, - rotate_speed * 2 * Time.deltaTime);
        // }       
        // // Acceleration
        // if ((Input.GetKey(KeyCode.I) || OVRInput.Get(OVRInput.Button.PrimaryThumbstickUp)) &&
        //     (Input.GetKey(KeyCode.I) || OVRInput.Get(OVRInput.Button.SecondaryThumbstickUp)))
        // {
        //     Spaceship.transform.Translate(Spaceship.transform.forward * (-1) * accelerate_speed * Time.deltaTime);
        // }          
        // if ((Input.GetKey(KeyCode.K) || OVRInput.Get(OVRInput.Button.PrimaryThumbstickDown)) &&
        //     (Input.GetKey(KeyCode.K) || OVRInput.Get(OVRInput.Button.SecondaryThumbstickDown)))
        // {
        //     Spaceship.transform.Translate(Spaceship.transform.forward * accelerate_speed * Time.deltaTime);
        // }           

        Vector3 rotationL = OVRInput.GetLocalControllerRotation(OVRInput.Controller.LTouch).eulerAngles;
        Vector3 rotationR = OVRInput.GetLocalControllerRotation(OVRInput.Controller.RTouch).eulerAngles;

        //  Rotation
        if (((rotationL.y > 270) && (rotationL.y < 340)) && ((rotationR.y > 270) && (rotationR.y < 340)))
        {
            Spaceship.transform.RotateAround(Spaceship.transform.position, Vector3.down, rotate_speed * Time.deltaTime);
            SteeringWheel.transform.RotateAround(SteeringWheel.transform.position, SteeringWheel.transform.forward, rotate_speed * 2 * Time.deltaTime);
        }         
        if (((rotationL.y > 20) && (rotationL.y < 90)) && ((rotationR.y > 20) && (rotationR.y < 90)))
        {
            Spaceship.transform.RotateAround(Spaceship.transform.position, Vector3.down, - rotate_speed * Time.deltaTime);
            SteeringWheel.transform.RotateAround(SteeringWheel.transform.position, SteeringWheel.transform.forward, - rotate_speed * 2 * Time.deltaTime);
        }       
        // Acceleration
        if (OVRInput.Get(OVRInput.Button.Two))
        {
            Spaceship.transform.Translate(Spaceship.transform.forward * (-1) * accelerate_speed * Time.deltaTime);
        }          
        if (OVRInput.Get(OVRInput.Button.One))
        {
            Spaceship.transform.Translate(Spaceship.transform.forward * accelerate_speed * Time.deltaTime);
        }                   

    }
}
