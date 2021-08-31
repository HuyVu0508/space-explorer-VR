using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class orbit : MonoBehaviour
{

    public GameObject pivot;  // the pivot to orbit around 
    public float speed;  // speed of rotating  
    public Vector3 axis; // axis of rotating

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        OrbitAround();  
    }


    // Orbitting method
    void OrbitAround()
    {
        transform.RotateAround(pivot.transform.position, axis, speed * Time.deltaTime);
    }
}
