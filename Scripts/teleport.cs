using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class teleport : MonoBehaviour
{
    public GameObject TeleportDest;

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.T) || OVRInput.GetUp(OVRInput.Button.Two))
        {
            transform.position = TeleportDest.transform.position;
        }             
    }
}
