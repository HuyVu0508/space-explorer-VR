using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class floating : MonoBehaviour
{

    public float max_moving_intensity; // Intensity of floating movement
    public float max_rotating_intensity; // Intensity of floating rotation

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        // Moving motion
        Vector3 moving_direction = new Vector3(UnityEngine.Random.Range(-1.0f,1.0f), UnityEngine.Random.Range(-1.0f,1.0f), UnityEngine.Random.Range(-1.0f,1.0f));
        float moving_intensity = UnityEngine.Random.Range(0,max_moving_intensity);
        GetComponent<Rigidbody>().AddForce(moving_direction *moving_intensity);    

        // Rotating motion
        Vector3 rotating_direction = new Vector3(UnityEngine.Random.Range(-1.0f,1.0f), UnityEngine.Random.Range(-1.0f,1.0f), UnityEngine.Random.Range(-1.0f,1.0f));
        float rotating_intensity = UnityEngine.Random.Range(0,max_rotating_intensity);
        GetComponent<Rigidbody>().AddTorque(rotating_intensity *rotating_direction);    
        // transform.Rotate(rotating_intensity *rotating_direction);
    }
}
