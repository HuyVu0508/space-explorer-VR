using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class return_to_shelf : MonoBehaviour
{
    public GameObject box;
    private Vector3 original_postition;
    private Quaternion original_rotation;
    public UnityEngine.UI.Text outText;

    // Start is called before the first frame update
    void Start()
    {
        original_postition = transform.position;
        original_rotation = transform.rotation;
    }

    // Update is called once per frame
    void Update()
    {

        // Check if object is at origianl position
        bool at_original;
        if ((transform.position == original_postition) && (transform.rotation == original_rotation)){
            at_original = true;
        }
        else{
            at_original = false;
        }

        // If object not grabbed
        if(!transform.GetComponent<OVRGrabbable>().isGrabbed){

            // If object at the box position
            if (close_to_box(transform.position)){
                transform.GetComponent<Rigidbody>().useGravity = true;

                Debug.Log("close_to_box!!!");
            }
            else{
                // If object not at original position
                if (!at_original){
                    transform.position = original_postition;
                    transform.rotation = original_rotation;
                    transform.GetComponent<Rigidbody>().useGravity = false;
                }
                Debug.Log("Not close_to_box!!!");
            }

            // if (!at_original){
            //     transform.position = original_postition;
            //     transform.rotation = original_rotation;
            //     transform.GetComponent<Rigidbody>().useGravity = false;    
            // }        

        }
    }

    // Method to check if object is close to box
    bool close_to_box(Vector3 object_position){

        Vector3 distance_vector = box.transform.position - object_position;
        float distance_value = distance_vector.magnitude;

        Debug.Log("distance_value: " + distance_value.ToString());
        outText.text = "distance_value: " + distance_value.ToString();

        // Check if object is close to box
        if (distance_value <= 6){
                return true;
            }
        else{
            return false;
        }    


        // // Box position on plane
        // Vector3 box_center = box.transform.position;
        // RectTransform rt = (RectTransform)box.transform;
        // // float box_width = rt.rect.width;
        // // float box_length = rt.rect.height;
        // float box_width = 0;
        // float box_length = 0;

        // return false;

        // // Check if object is close to box
        // if ((Mathf.Abs(box_center.x - object_position.x) <= box_width/2) &&
        //     (Mathf.Abs(box_center.z - object_position.z) <= box_length/2)){
        //         return true;
        //     }
        // else{
        //     return false;
        // }    

    }

}
