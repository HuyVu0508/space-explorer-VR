using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class canvas : MonoBehaviour
{
    private bool canvas_enabled;
    private GameObject canvasObject;

    // Start is called before the first frame update
    void Start()
    {
        canvasObject = transform.Find("Canvas").gameObject;
        canvas_enabled = true;
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.I) || OVRInput.GetUp(OVRInput.Button.Two)){
            if (canvas_enabled){
                canvasObject.SetActive(false);
                canvas_enabled = false;
            }
            else{
                canvasObject.SetActive(true);
                canvas_enabled = true;            
            }
        }
    }
}
