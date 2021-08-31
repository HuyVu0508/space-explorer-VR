using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class turnon_light_auto : MonoBehaviour
{
    public GameObject light_source;
    private bool turned_on;

    // Start is called before the first frame update
    void Start()
    {
        turned_on = false;
    }

    // Update is called once per frame
    void Update()
    {
        Vector3 light_ray_vector =  light_source.transform.position - transform.position;
        RaycastHit[] hits;
        hits = Physics.RaycastAll(transform.position, light_ray_vector, light_ray_vector.magnitude);

        // Check if ray hits an object different from front_glass or OVRPlayerController
        bool flag = false;
        for (int i = 0; i < hits.Length; i++)
        {
            RaycastHit hit = hits[i];
            if ((hit.collider.gameObject.name != "OVRPlayerController") && (hit.collider.gameObject.name != "front_glass"))
            {
                flag = true;
                break;
            }
        }



        // If light_ray hit collider and cannot come to spaceship light
        if(flag==true)
        {
            // If spaceship light is not turned on 
            if(turned_on==false)
            {
                GetComponent<Light>().enabled = true;
                transform.GetChild(0).gameObject.SetActive(true);
                transform.GetChild(1).gameObject.SetActive(false);
                turned_on = true;
            } 
        }
        else   // If light_ray is not blocked and come to spaceship light
        {
            // If spaceship light is not turned off 
            if(turned_on==true)
            {
                GetComponent<Light>().enabled = false;
                transform.GetChild(0).gameObject.SetActive(false);
                transform.GetChild(1).gameObject.SetActive(true);
                turned_on = false;
            }             
        }
    }
}
