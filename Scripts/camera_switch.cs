using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class camera_switch : MonoBehaviour
{
    public GameObject cameraOne;
    public GameObject cameraTwo;


    // Start is called before the first frame update
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.C) || OVRInput.GetUp(OVRInput.Button.Three))
        {
            switch_camera_OS();
        }             
        
        if (Input.GetKeyDown(KeyCode.V) || OVRInput.GetUp(OVRInput.Button.Four))
        {
            switch_camera_IS();
        }              
    }

    // Methods for changing camera 
    void switch_camera_OS()
    {   
        cameraTwo.GetComponent<Camera>().enabled = true;
        cameraOne.transform.Find("OVRCameraRig").Find("TrackingSpace").gameObject.SetActive(false);
    }
    void switch_camera_IS()
    {
        cameraTwo.GetComponent<Camera>().enabled = false;
        cameraOne.transform.Find("OVRCameraRig").Find("TrackingSpace").gameObject.SetActive(true);
    }    
}
