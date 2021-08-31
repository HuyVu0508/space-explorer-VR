using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class gravity_switch : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.G) || OVRInput.GetUp(OVRInput.Button.One))
        {
            switch_gravity();
        }               
    }

    void switch_gravity()
    {
        if (GetComponent<Rigidbody>().useGravity == true)
        {
            GetComponent<Rigidbody>().useGravity = false;
            GetComponent<Rigidbody>().AddForce((-1) * Physics.gravity/2);
        }
        else
        {
            GetComponent<Rigidbody>().useGravity = true;    
        };
    }
}
